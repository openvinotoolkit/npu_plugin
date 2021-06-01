//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "vpux/compiler/compiler.hpp"

#include "vpux/compiler/backend/VPUIP.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/network_description.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIP/passes.hpp"
#include "vpux/compiler/frontend/IE.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/pipelines.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_utils.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

#include <llvm/Support/Regex.h>

#include <cpp/ie_cnn_network.h>
#include <description_buffer.hpp>

#include <algorithm>

using namespace vpux;
using namespace InferenceEngine;

namespace {

//
// getLogLevel
//

LogLevel getLogLevel(const VPUXConfig& config) {
    switch (config.logLevel()) {
    case vpu::LogLevel::Fatal:
        return LogLevel::Fatal;
    case vpu::LogLevel::Error:
        return LogLevel::Error;
    case vpu::LogLevel::Warning:
        return LogLevel::Warning;
    case vpu::LogLevel::Info:
        return LogLevel::Info;
    case vpu::LogLevel::Debug:
        return LogLevel::Debug;
    case vpu::LogLevel::Trace:
        return LogLevel::Trace;
    default:
        return LogLevel::None;
    }
}

//
// getArchKind
//

VPUIP::ArchKind getArchKind(const VPUXConfig& config) {
    switch (config.platform()) {
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO:
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400_A0:
        return VPUIP::ArchKind::VPU3400_A0;
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400:
        return VPUIP::ArchKind::VPU3400;
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700:
        return VPUIP::ArchKind::VPU3700;
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3900:
        return VPUIP::ArchKind::VPU3900;
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720:
        return VPUIP::ArchKind::VPU3720;
    default:
        VPUX_THROW("Unsupported VPUX platform");
    }
}

VPUIP::CompilationMode getCompilationMode(const VPUXConfig& config) {
    const auto parsed = VPUIP::symbolizeCompilationMode(config.compilationMode());
    VPUX_THROW_UNLESS(parsed.hasValue(), "Unsupported compilation mode '{0}'", config.compilationMode());
    return parsed.getValue();
}

//
// TimingLogger
//

class TimingLogger final : public mlir::PassManager::PassTimingConfig {
public:
    TimingLogger(): _colorStream(Logger::getLevelStream(LogLevel::Info)) {
    }

public:
    void printTiming(PrintCallbackFn printCallback) final {
        printCallback(_colorStream.get());
    }

private:
    llvm::WithColor _colorStream;
};

//
// DeveloperConfig
//

class DeveloperConfig final {
public:
    DeveloperConfig();
    ~DeveloperConfig();

    void setup(mlir::PassManager& pm, Logger log) const;

    bool useSharedConstants() const {
        return _crashReproducerFile.empty() && _irPrintingFilter.empty();
    }

private:
    std::string _crashReproducerFile;
    bool _localReproducer = true;

    std::string _irPrintingFilter;
    std::string _irPrintingFile;
    bool _printFullIR = false;
    bool _printFullConstant = false;

    llvm::raw_ostream* _stream = nullptr;
    std::unique_ptr<llvm::Regex> _filter;
    std::unique_ptr<llvm::WithColor> _colorStream;
    std::unique_ptr<llvm::raw_fd_ostream> _fileStream;
};

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

void parseEnv(StringRef envVarName, std::string& var) {
    if (const auto env = std::getenv(envVarName.data())) {
        var = env;
    }
}

void parseEnv(StringRef envVarName, bool& var) {
    if (const auto env = std::getenv(envVarName.data())) {
        var = std::stoi(env);
    }
}

#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

DeveloperConfig::DeveloperConfig() {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    parseEnv("IE_VPUX_CRASH_REPRODUCER_FILE", _crashReproducerFile);
    parseEnv("IE_VPUX_GEN_LOCAL_REPRODUCER", _localReproducer);

    parseEnv("IE_VPUX_IR_PRINTING_FILTER", _irPrintingFilter);
    parseEnv("IE_VPUX_IR_PRINTING_FILE", _irPrintingFile);
    parseEnv("IE_VPUX_PRINT_FULL_IR", _printFullIR);
    parseEnv("IE_VPUX_PRINT_FULL_CONSTANT", _printFullConstant);
#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

    if (!_irPrintingFilter.empty()) {
        _filter = std::make_unique<llvm::Regex>(_irPrintingFilter, llvm::Regex::IgnoreCase);

        std::string regexErr;
        if (!_filter->isValid(regexErr)) {
            VPUX_THROW("Invalid regular expression '{0}' : {1}", _irPrintingFilter, regexErr);
        }

        if (_irPrintingFile.empty()) {
            _colorStream = std::make_unique<llvm::WithColor>(Logger::getLevelStream(LogLevel::Trace));

            _stream = &_colorStream->get();
        } else {
            std::error_code err;
            _fileStream = std::make_unique<llvm::raw_fd_ostream>(_irPrintingFile, err);
            if (err) {
                VPUX_THROW("Failed to open file '{0}' for write : {1}", _irPrintingFile, err.message());
            }

            _stream = _fileStream.get();
        }
    }
}

DeveloperConfig::~DeveloperConfig() {
    if (_stream != nullptr) {
        _stream->flush();
    }
}

void DeveloperConfig::setup(mlir::PassManager& pm, Logger log) const {
    // Pass timing

    if (log.isActive(LogLevel::Info)) {
        pm.enableTiming(std::make_unique<TimingLogger>());
    }

    // Crash reproducer

    if (!_crashReproducerFile.empty()) {
        pm.enableCrashReproducerGeneration(_crashReproducerFile, _localReproducer);
    }

    // IR printing

    if (_filter != nullptr) {
        const bool printAfterOnlyOnChange = false;

        const auto shouldPrintBeforePass = [&](mlir::Pass*, mlir::Operation*) {
            return false;
        };
        const auto shouldPrintAfterPass = [&](mlir::Pass* pass, mlir::Operation*) {
            return _filter->match(pass->getName()) || _filter->match(pass->getArgument());
        };

        if (_printFullIR) {
            pm.getContext()->disableMultithreading();
        }

        mlir::OpPrintingFlags flags = mlir::OpPrintingFlags();
        if (!_printFullConstant) {
            flags.elideLargeElementsAttrs();
        }

        pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, _printFullIR, printAfterOnlyOnChange, *_stream,
                            flags);
    }
}

}  // namespace

//
// CompilerImpl::query
//

InferenceEngine::QueryNetworkResult vpux::CompilerImpl::query(const InferenceEngine::CNNNetwork& /*network*/,
                                                              const vpux::VPUXConfig& /*config*/) {
    InferenceEngine::QueryNetworkResult result;
    return result;
}

//
// CompilerImpl::compile
//

std::shared_ptr<INetworkDescription> vpux::CompilerImpl::compile(const std::shared_ptr<ngraph::Function>& func,
                                                                 const std::string&, const InputsDataMap& inputsInfo,
                                                                 const OutputsDataMap& outputsInfo,
                                                                 const VPUXConfig& config) {
    //
    // Parse config options
    //

    DeveloperConfig devConf;

    const auto compilationMode = getCompilationMode(config);
    const bool sharedConstants = devConf.useSharedConstants();

    //
    // Initialize compiler
    //

    Logger log("vpux-compiler", getLogLevel(config));

    mlir::DialectRegistry registry;
    registerDialects(registry);

    mlir::MLIRContext ctx(registry);
    addLogging(ctx, log);

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    addLogging(pm, log);
    devConf.setup(pm, log);

    //
    // Build compilation pipeline
    //

    pm.addPass(createSetCompileParamsPass(getArchKind(config), compilationMode, log.nest()));

    if (compilationMode == VPUIP::CompilationMode::ReferenceSW) {
        buildReferenceModePipeline(pm, log.nest());
    } else if (compilationMode == VPUIP::CompilationMode::ReferenceHW) {
        buildHardwareModePipeline(pm, log.nest());
    } else {
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }

    //
    // Process the network
    //

    CNNNetwork cnnNet(func);
    for (const auto& p : inputsInfo) {
        cnnNet.getInputsInfo().at(p.first)->setPrecision(p.second->getPrecision());
        cnnNet.getInputsInfo().at(p.first)->setLayout(p.second->getLayout());
    }
    for (const auto& p : outputsInfo) {
        cnnNet.getOutputsInfo().at(p.first)->setPrecision(p.second->getPrecision());
        cnnNet.getOutputsInfo().at(p.first)->setLayout(p.second->getLayout());
    }

    auto module = IE::importNetwork(&ctx, cnnNet, sharedConstants, log.nest());

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module.get())), "Compilation failed");

    const auto blob = VPUIP::exportToBlob(module.get(), log);

    //
    // Return compiled blob and meta-data
    //

    std::vector<char> compiledNetwork(blob.size());
    std::copy_n(reinterpret_cast<const char*>(blob.data()), blob.size(), compiledNetwork.data());

    return std::make_shared<VPUIP::NetworkDescription>(std::move(compiledNetwork));
}

//
// CompilerImpl::parse
//

std::shared_ptr<vpux::INetworkDescription> vpux::CompilerImpl::parse(const std::vector<char>& compiledNetwork,
                                                                     const vpux::VPUXConfig&, const std::string&) {
    return std::make_shared<VPUIP::NetworkDescription>(compiledNetwork);
}

//
// CompilerImpl::getSupportedOptions
//

std::unordered_set<std::string> vpux::CompilerImpl::getSupportedOptions() {
    return {};
}

//
// CreateVPUXCompiler
//

INFERENCE_PLUGIN_API(void)
CreateVPUXCompiler(std::shared_ptr<ICompiler>& compiler) {
    compiler = std::make_shared<CompilerImpl>();
}
