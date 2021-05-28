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
    explicit TimingLogger(Logger log): _log(log) {
    }

public:
    void printTiming(PrintCallbackFn printCallback) final;

private:
    Logger _log;
};

void TimingLogger::printTiming(PrintCallbackFn printCallback) {
    std::string timingLog;
    llvm::raw_string_ostream stream(timingLog);
    printCallback(stream);

    splitStringList(timingLog, '\n', [this](StringRef line) {
        _log.info("{0}", line);
    });
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

    auto compilationMode = getCompilationMode(config);

    // TODO: move this to config class
    bool enablePassVerifier = true;
    std::string crashReproducerFile;
    bool localReproducer = true;
    Optional<llvm::Regex> irPrintingFilter;
    bool printFullIR = false;
    bool printFullConstant = false;

#ifdef VPUX_DEVELOPER_BUILD
    if (const auto env = std::getenv("IE_VPUX_ENABLE_PASS_VERIFIER")) {
        enablePassVerifier = std::stoi(env);
    }
    if (const auto env = std::getenv("IE_VPUX_CRASH_REPRODUCER_FILE")) {
        crashReproducerFile = env;
    }
    if (const auto env = std::getenv("IE_VPUX_GEN_LOCAL_REPRODUCER")) {
        localReproducer = std::stoi(env);
    }
    if (const auto env = std::getenv("IE_VPUX_IR_PRINTING_FILTER")) {
        const StringRef filter(env);

        if (!filter.empty()) {
            irPrintingFilter = llvm::Regex(filter, llvm::Regex::IgnoreCase);

            std::string regexErr;
            if (!irPrintingFilter->isValid(regexErr)) {
                VPUX_THROW("Invalid regural expression '{0}' : {1}", filter, regexErr);
            }
        }
    }
    if (const auto env = std::getenv("IE_VPUX_PRINT_FULL_IR")) {
        printFullIR = std::stoi(env);
    }

    if (const auto env = std::getenv("IE_VPUX_COMPILATION_MODE")) {
        const auto parsed = VPUIP::symbolizeCompilationMode(env);
        VPUX_THROW_UNLESS(parsed.hasValue(), "Unsupported compilation mode '{0}'", env);
        compilationMode = parsed.getValue();
    }

    if (const auto env = std::getenv("IE_VPUX_PRINT_FULL_CONSTANT")) {
        printFullConstant = std::stoi(env);
    }
#endif

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
    pm.enableVerifier(enablePassVerifier);
    if (!crashReproducerFile.empty()) {
        pm.enableCrashReproducerGeneration(crashReproducerFile, localReproducer);
    }
    if (log.isActive(LogLevel::Info)) {
        pm.enableTiming(std::make_unique<TimingLogger>(log));
    }
    if (irPrintingFilter.hasValue()) {
        const auto shouldPrintForPass = [&](mlir::Pass* pass, mlir::Operation*) {
            return irPrintingFilter->match(pass->getName()) || irPrintingFilter->match(pass->getArgument());
        };

        auto colorStream = Logger::getLevelStream(LogLevel::Trace);
        auto& stream = colorStream.get();

        if (printFullIR) {
            ctx.disableMultithreading();
        }

        mlir::OpPrintingFlags flags = mlir::OpPrintingFlags();
        if (!printFullConstant) {
            flags.elideLargeElementsAttrs();
        }
        pm.enableIRPrinting(shouldPrintForPass, shouldPrintForPass, printFullIR, false, stream, flags);
    }

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

    const bool sharedConstants = crashReproducerFile.empty() && !irPrintingFilter.hasValue();
    auto module = IE::importNetwork(&ctx, cnnNet, sharedConstants, log.nest());

    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module.get())), "Compilation failed");

    if (irPrintingFilter.hasValue()) {
        auto& stream = Logger::getBaseStream();
        stream.flush();
    }

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
