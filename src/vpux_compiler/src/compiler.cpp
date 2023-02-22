//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux/compiler/compiler.hpp"

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"

#include "vpux/compiler/conversion.hpp"
#include "vpux/compiler/dialect/ELF/export.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/IERT/ops.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPU/passes.hpp"
#include "vpux/compiler/dialect/VPUIP/network_description.hpp"
#include "vpux/compiler/dialect/VPUIP/ops.hpp"
#include "vpux/compiler/dialect/VPUIPRegMapped/network_description.hpp"
#include "vpux/compiler/frontend/IE.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/pipelines.hpp"
#include "vpux/compiler/utils/dot_printer.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/helper_macros.hpp"
#include "vpux/utils/core/optional.hpp"
#include "vpux/utils/core/string_utils.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/Timing.h>

#include <llvm/Support/Regex.h>
#include <llvm/Support/raw_ostream.h>

#include <cpp/ie_cnn_network.h>
#include <description_buffer.hpp>
#include <device_helpers.hpp>
#include <ie_ngraph_utils.hpp>
#include <transformations/utils/utils.hpp>

#include <algorithm>

#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

#include "vpux/compiler/core/developer_build_utils.hpp"

#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

using namespace vpux;
using namespace InferenceEngine;

namespace {

//
// getArchKind
//

VPU::ArchKind getArchKind(const Config& config) {
    auto platform = config.get<PLATFORM>();
    if (platform == InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR) {
        platform = utils::getPlatformByEMUDeviceName(config.get<DEVICE_ID>());
    }

    switch (platform) {
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO:
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR:
        return VPU::ArchKind::UNKNOWN;
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400:
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700:
        return VPU::ArchKind::VPUX30XX;
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3800:
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3900:
        return VPU::ArchKind::VPUX311X;
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720:
        return VPU::ArchKind::VPUX37XX;
    default:
        VPUX_THROW("Unsupported VPUX platform");
    }
}

//
// getCompilationMode
//

VPU::CompilationMode getCompilationMode(const Config& config) {
    if (!config.has<COMPILATION_MODE>()) {
        return VPU::CompilationMode::DefaultHW;
    }

    const auto parsed = VPU::symbolizeCompilationMode(config.get<COMPILATION_MODE>());
    VPUX_THROW_UNLESS(parsed.hasValue(), "Unsupported compilation mode '{0}'", config.get<COMPILATION_MODE>());
    return parsed.getValue();
}

//
// getDDRHeapSize
//

Optional<int> getDDRHeapSize(const Config& config) {
    if (config.has<DDR_HEAP_SIZE_MB>()) {
        return checked_cast<int>(config.get<DDR_HEAP_SIZE_MB>());
    }

    return 500;
}

//
// getNumberOfDPUGroups
//

Optional<int> getNumberOfDPUGroups(const Config& config) {
    if (config.has<DPU_GROUPS>()) {
        return checked_cast<int>(config.get<DPU_GROUPS>());
    }

    switch (config.get<PLATFORM>()) {
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720: {
        switch (config.get<PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
        case ov::hint::PerformanceMode::UNDEFINED:
        case ov::hint::PerformanceMode::LATENCY:
        default:
            return checked_cast<int>(VPU::getMaxDPUClusterNum(getArchKind(config)));
        }
        break;
    }
    default: {
        switch (config.get<PERFORMANCE_HINT>()) {
        case ov::hint::PerformanceMode::THROUGHPUT:
            return 1;
        case ov::hint::PerformanceMode::LATENCY:
        case ov::hint::PerformanceMode::UNDEFINED:
        default:
            return checked_cast<int>(VPU::getMaxDPUClusterNum(getArchKind(config)));
        }
        break;
    }
    }
}

//
// getNumberOfDMAPorts
//

Optional<int> getNumberOfDMAPorts(const Config& config) {
    if (config.has<DMA_ENGINES>()) {
        return checked_cast<int>(config.get<DMA_ENGINES>());
    }

    switch (config.get<PERFORMANCE_HINT>()) {
    case ov::hint::PerformanceMode::THROUGHPUT:
        return 1;
    case ov::hint::PerformanceMode::LATENCY:
    case ov::hint::PerformanceMode::UNDEFINED:
    default:
        return checked_cast<int>(VPU::getMaxDMAPorts(getArchKind(config)));
    }
}

//
// DeveloperConfig
//

class DeveloperConfig final {
public:
    explicit DeveloperConfig(Logger log);
    ~DeveloperConfig();

    void setup(mlir::DefaultTimingManager& tm) const;
    void setup(mlir::PassManager& pm) const;

    bool useSharedConstants() const {
        return _crashReproducerFile.empty() && _irPrintingFilter.empty();
    }

private:
    Logger _log;

    std::string _crashReproducerFile;
    bool _localReproducer = true;

    std::string _irPrintingFilter;
    std::string _irPrintingFile;
    std::string _irPrintingOrderStr;
    bool _printFullIR = false;
    bool _printFullConstant = false;
    bool _printDebugInfo = false;
    std::string _printDotOptions;

    llvm::raw_ostream* _timingStream = nullptr;

    std::unique_ptr<llvm::Regex> _irDumpFilter;
    std::unique_ptr<llvm::raw_fd_ostream> _irDumpFile;
    llvm::raw_ostream* _irDumpStream = nullptr;
    IRPrintingOrder _irPrintingOrder = IRPrintingOrder::AFTER;
};

DeveloperConfig::DeveloperConfig(Logger log): _log(log) {
#if defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)
    parseEnv("IE_VPUX_CRASH_REPRODUCER_FILE", _crashReproducerFile);
    parseEnv("IE_VPUX_GEN_LOCAL_REPRODUCER", _localReproducer);

    parseEnv("IE_VPUX_IR_PRINTING_FILTER", _irPrintingFilter);
    parseEnv("IE_VPUX_IR_PRINTING_FILE", _irPrintingFile);
    parseEnv("IE_VPUX_IR_PRINTING_ORDER", _irPrintingOrderStr);
    parseEnv("IE_VPUX_PRINT_FULL_IR", _printFullIR);
    parseEnv("IE_VPUX_PRINT_FULL_CONSTANT", _printFullConstant);
    parseEnv("IE_VPUX_PRINT_DEBUG_INFO", _printDebugInfo);

    parseEnv("IE_VPUX_PRINT_DOT", _printDotOptions);
#endif  // defined(VPUX_DEVELOPER_BUILD) || !defined(NDEBUG)

    if (_log.isActive(LogLevel::Info)) {
        _timingStream = &Logger::getBaseStream();
    }

    if (!_irPrintingOrderStr.empty()) {
        auto orderString = _irPrintingOrderStr;
        std::transform(orderString.begin(), orderString.end(), orderString.begin(), [](unsigned char c) {
            return std::toupper(c);
        });
        if (orderString == "BEFORE") {
            _irPrintingOrder = IRPrintingOrder::BEFORE;
        } else if (orderString == "AFTER") {
            _irPrintingOrder = IRPrintingOrder::AFTER;
        } else if (orderString == "BEFORE_AFTER") {
            _irPrintingOrder = IRPrintingOrder::BEFORE_AFTER;
        } else {
            VPUX_THROW("Invalid IR printing order: {0}.\nValid cases are: before, after and before_after. They are not "
                       "case-sensitive.\nExample: IE_VPUX_IR_PRINTING_ORDER=Before",
                       _irPrintingOrderStr);
        }
    }

    if (!_irPrintingFilter.empty()) {
        _irDumpFilter = std::make_unique<llvm::Regex>(_irPrintingFilter, llvm::Regex::IgnoreCase);

        std::string regexErr;
        if (!_irDumpFilter->isValid(regexErr)) {
            VPUX_THROW("Invalid regular expression '{0}' : {1}", _irPrintingFilter, regexErr);
        }

        if (_irPrintingFile.empty()) {
            _irDumpStream = &vpux::Logger::getBaseStream();
        } else {
            std::error_code err;
            _irDumpFile = std::make_unique<llvm::raw_fd_ostream>(_irPrintingFile, err);
            if (err) {
                VPUX_THROW("Failed to open file '{0}' for write : {1}", _irPrintingFile, err.message());
            }

            _irDumpStream = _irDumpFile.get();
        }
    }
}

DeveloperConfig::~DeveloperConfig() {
    if (_timingStream != nullptr) {
        _timingStream->flush();
    }

    if (_irDumpStream != nullptr) {
        _irDumpStream->flush();
    }
}

void DeveloperConfig::setup(mlir::DefaultTimingManager& tm) const {
    if (_timingStream == nullptr) {
        tm.setEnabled(false);
    } else {
        tm.setEnabled(true);
        tm.setDisplayMode(mlir::DefaultTimingManager::DisplayMode::Tree);
        tm.setOutput(*_timingStream);
    }
}

void DeveloperConfig::setup(mlir::PassManager& pm) const {
    // Crash reproducer

    if (!_crashReproducerFile.empty()) {
        if (_localReproducer) {
            pm.getContext()->disableMultithreading();
        }

        pm.enableCrashReproducerGeneration(_crashReproducerFile, _localReproducer);
    }

    // IR printing

    if (_irDumpFilter != nullptr) {
        const bool printAfterOnlyOnChange = false;
        const bool printAfterOnlyOnFailure = false;

        const auto shouldPrintBeforePass = [&](mlir::Pass* pass, mlir::Operation*) {
            return (_irDumpFilter->match(pass->getName()) || _irDumpFilter->match(pass->getArgument())) &&
                   (_irPrintingOrder == IRPrintingOrder::BEFORE || _irPrintingOrder == IRPrintingOrder::BEFORE_AFTER);
        };
        const auto shouldPrintAfterPass = [&](mlir::Pass* pass, mlir::Operation*) {
            return (_irDumpFilter->match(pass->getName()) || _irDumpFilter->match(pass->getArgument())) &&
                   (_irPrintingOrder == IRPrintingOrder::AFTER || _irPrintingOrder == IRPrintingOrder::BEFORE_AFTER);
        };

        if (_printFullIR) {
            pm.getContext()->disableMultithreading();
        }

        mlir::OpPrintingFlags flags;
        if (!_printFullConstant) {
            flags.elideLargeElementsAttrs();
        }
        if (_printDebugInfo) {
            flags.enableDebugInfo(true);
        }

        pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, _printFullIR, printAfterOnlyOnChange,
                            printAfterOnlyOnFailure, *_irDumpStream, flags);
    }

    // Dot printing
    if (!_printDotOptions.empty()) {
        addDotPrinter(pm, _printDotOptions);
    }
}

}  // namespace

//
// CompilerImpl::query
//

InferenceEngine::QueryNetworkResult vpux::CompilerImpl::query(const InferenceEngine::CNNNetwork& network,
                                                              const Config& config) {
    Logger log("vpux-compiler", config.get<LOG_LEVEL>());
    InferenceEngine::QueryNetworkResult result;
    const std::string plugin_name = DEVICE_NAME;

    const auto arch = getArchKind(config);

    DeveloperConfig devConf(log);
    mlir::DefaultTimingManager tm;
    devConf.setup(tm);
    auto rootTiming = tm.getRootScope();

    std::vector<vpux::PreProcessInfo> preProcInfo;
    auto supportedLayers = IE::queryNetwork(network, preProcInfo, rootTiming, arch, log);

    for (auto&& layerName : supportedLayers) {
        result.supportedLayersMap.emplace(layerName, plugin_name);
    }

    return result;
}

//
// CompilerImpl::compile
//

namespace {

void buildPipeline(mlir::PassManager& pm, const Config& config, mlir::TimingScope& rootTiming, Logger log,
                   const bool inAndOutFp16, bool floatInputPrecision) {
    auto buildTiming = rootTiming.nest("Build compilation pipeline");

    const auto archKind = getArchKind(config);
    const auto compilationMode = getCompilationMode(config);
    const auto enableProfiling = config.get<PERF_COUNT>();
    const auto numOfDPUGroups = getNumberOfDPUGroups(config);
    const auto numOfDMAPorts = getNumberOfDMAPorts(config);
    const auto ddrHeapSize = getDDRHeapSize(config);

    pm.addPass(VPU::createInitCompilerPass(archKind, compilationMode, numOfDPUGroups, numOfDMAPorts, ddrHeapSize,
                                           log.nest()));

    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        const auto options = ReferenceSWOptions::createFromString(config.get<COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->enableProfiling = enableProfiling;
        if (config.get<PLATFORM>() == InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR) {
            buildEMUReferenceSWModePipeline(pm, *options, log.nest());
        } else {
            buildReferenceSWModePipeline(pm, *options, log.nest());
        }
    } else if (compilationMode == VPU::CompilationMode::ReferenceHW) {
        const auto options = ReferenceHWOptions::createFromString(config.get<COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->enableProfiling = enableProfiling;
        if (config.get<PLATFORM>() == InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR) {
            buildEMUReferenceHWModePipeline(pm, *options, log.nest());
        } else {
            buildReferenceHWModePipeline(pm, *options, log.nest());
        }
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        const auto options = DefaultHWOptions::createFromString(config.get<COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "buildPipeline failed to parse COMPILATION_MODE_PARAMS");
        options->enableProfiling = enableProfiling;

        if (archKind == VPU::ArchKind::VPUX37XX) {
            options->enableConvertAvgPoolToDWConv = false;
            options->enableHandleAsymmetricStrides = false;
        }
        // floatInputPrecision:
        // In case user passes -ip fp16/fp32 and enables FORCE_HOST_QUANTIZATION feature
        // we perform quantization on host.
        options->forceHostInputQuantization = config.get<FORCE_HOST_QUANTIZATION>() && floatInputPrecision;

        if (archKind == VPU::ArchKind::VPUX30XX) {
            options->forceHostPrecisionLayoutConversion = config.get<FORCE_HOST_PRECISION_LAYOUT_CONVERSION>();
            if (inAndOutFp16) {
#if defined(_WIN32)
                options->forceHostPrecisionLayoutConversion = false;
                options->enableUseUserPrecision = false;
                options->enableUseUserLayout = false;
#endif
            }
        }

        if (config.get<PLATFORM>() == InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR) {
            buildEMUDefaultHWModePipeline(pm, *options, log.nest());
        } else {
            buildDefaultHWModePipeline(pm, *options, log.nest());
        }
    } else {
        VPUX_THROW("Unsupported compilation mode '{0}'", compilationMode);
    }
    if (config.get<USE_ELF_COMPILER_BACKEND>()) {
        buildLowerVPUIP2ELFPipeline(pm, log.nest());
    }
}

CNNNetwork prepareNetwork(const std::shared_ptr<ngraph::Function>& func, const InputsDataMap& inputsInfo,
                          const OutputsDataMap& outputsInfo, mlir::TimingScope& rootTiming) {
    auto prepareTiming = rootTiming.nest("Prepare network");

    CNNNetwork cnnNet(func);

    for (const auto& p : inputsInfo) {
        cnnNet.getInputsInfo().at(p.first)->setPrecision(p.second->getPrecision());
        cnnNet.getInputsInfo().at(p.first)->setLayout(p.second->getLayout());
    }
    for (const auto& p : outputsInfo) {
        cnnNet.getOutputsInfo().at(p.first)->setPrecision(p.second->getPrecision());
        cnnNet.getOutputsInfo().at(p.first)->setLayout(p.second->getLayout());
    }

    return cnnNet;
}

auto importNetwork(mlir::MLIRContext* ctx, CNNNetwork cnnNet, const DeveloperConfig& devConf,
                   std::vector<vpux::PreProcessInfo>& preProcInfo, mlir::TimingScope& rootTiming, bool enableProfiling,
                   bool stubLayers, vpux::VPU::ArchKind arch, Logger log) {
    auto importTiming = rootTiming.nest("Import network");
    return IE::importNetwork(ctx, cnnNet, preProcInfo, devConf.useSharedConstants(), importTiming, enableProfiling,
                             stubLayers, arch, log.nest());
}

void compileNetwork(mlir::ModuleOp module, mlir::PassManager& pm, mlir::TimingScope& rootTiming) {
    auto compileTiming = rootTiming.nest("Compile network");
    pm.enableTiming(compileTiming);
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");
}

auto exportToELF(mlir::ModuleOp module, const std::vector<vpux::PreProcessInfo>& preprocessInfo,
                 const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                 const std::vector<std::shared_ptr<const ov::Node>>& results) {
    return vpux::ELF::exportToELF(module, preprocessInfo, parameters, results);
}

bool isIR10(const ov::Model& model) {
    const auto& rtInfo = model.get_rt_info();
    const auto it = rtInfo.find("version");
    if (it != rtInfo.end()) {
        const int64_t irVersion = it->second.as<int64_t>();
        return irVersion == 10;
    }
    return false;
}

std::vector<std::shared_ptr<const ov::Node>> buildOVParams(const std::shared_ptr<ov::Model>& model,
                                                           const InputsDataMap& inputsInfo) {
    std::vector<std::shared_ptr<const ov::Node>> constParams;
    VPUX_THROW_WHEN(model == nullptr, "Null OV model");

    // Here we decide whether we need to add operation_names as tensor names for
    // getInputs / getOutputs. Since these functions are designed to be used in new API only
    // always need to add operation names for IR v10
    const auto addOpNames = isIR10(*model);

    for (const auto& param : model->get_parameters()) {
        auto newParam = ov::as_type_ptr<ov::op::v0::Parameter>(param->copy_with_new_inputs({}));
        newParam->set_friendly_name(param->get_friendly_name());
        if (addOpNames)
            newParam->output(0).get_tensor().add_names({newParam->get_friendly_name()});
        // WA: use CNNNetwork's precisions since plugins sometimes override their precisions
        // after transformation pipeline is run
        newParam->set_element_type(InferenceEngine::details::convertPrecision(
                inputsInfo.at(newParam->get_friendly_name())->getPrecision()));
        newParam->validate_and_infer_types();
        constParams.emplace_back(newParam);
    }

    return constParams;
}

std::vector<std::shared_ptr<const ov::Node>> buildOVResults(const std::shared_ptr<ov::Model>& model,
                                                            const OutputsDataMap& outputsInfo) {
    std::vector<std::shared_ptr<const ov::Node>> constResults;
    VPUX_THROW_WHEN(model == nullptr, "Null OV model");

    // Here we decide whether we need to add operation_names as tensor names for
    // getInputs / getOutputs. Since these functions are designed to be used in new API only
    // always need to add operation names for IR v10
    const auto addOpNames = isIR10(*model);

    for (const auto& result : model->get_results()) {
        auto fakeParam = std::make_shared<ov::op::v0::Parameter>(result->get_output_element_type(0),
                                                                 result->get_output_partial_shape(0));
        const std::string paramName = ngraph::op::util::create_ie_output_name(result->input_value(0));
        fakeParam->set_friendly_name(paramName);
        fakeParam->set_element_type(
                InferenceEngine::details::convertPrecision(outputsInfo.at(paramName)->getPrecision()));
        fakeParam->validate_and_infer_types();
        auto newResult = result->copy_with_new_inputs({fakeParam});
        newResult->set_friendly_name(result->get_friendly_name());
        if (addOpNames) {
            newResult->output(0).get_tensor().add_names({fakeParam->get_friendly_name()});
        }
        constResults.emplace_back(newResult);
    }

    return constResults;
}

}  // namespace

std::shared_ptr<INetworkDescription> vpux::CompilerImpl::compile(const std::shared_ptr<ngraph::Function>& func,
                                                                 const std::string&, const InputsDataMap& inputsInfo,
                                                                 const OutputsDataMap& outputsInfo,
                                                                 const Config& config) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compile");
    Logger log("vpux-compiler", config.get<LOG_LEVEL>());

    DeveloperConfig devConf(log);

    mlir::DefaultTimingManager tm;
    devConf.setup(tm);

    mlir::DialectRegistry registry;
    registerDialects(registry);

    bool enableDummyOpReplacement = false;

    bool inAndOutFp16 = false;
    bool floatInputPrecision = false;
    // TODO: needs refactoring. Ticket: E#50937
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        const auto options = ReferenceSWOptions::createFromString(config.get<COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "failed to parse COMPILATION_MODE_PARAMS");
        enableDummyOpReplacement = options->enableDummyOpReplacement;
    } else if (compilationMode == VPU::CompilationMode::ReferenceHW) {
        const auto options = ReferenceHWOptions::createFromString(config.get<COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "failed to parse COMPILATION_MODE_PARAMS");
        enableDummyOpReplacement = options->enableDummyOpReplacement;
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        const auto options = DefaultHWOptions::createFromString(config.get<COMPILATION_MODE_PARAMS>());
        VPUX_THROW_UNLESS(options != nullptr, "failed to parse COMPILATION_MODE_PARAMS");
        enableDummyOpReplacement = options->enableDummyOpReplacement;

        bool inFp16 = false, outFp16 = false;
        for (const auto& p : inputsInfo) {
            inFp16 = p.second->getPrecision() == InferenceEngine::Precision::FP16;
            floatInputPrecision = p.second->getPrecision() == InferenceEngine::Precision::FP16 ||
                                  p.second->getPrecision() == InferenceEngine::Precision::FP32;
        }
        for (const auto& p : outputsInfo) {
            outFp16 = p.second->getPrecision() == InferenceEngine::Precision::FP16;
        }
        inAndOutFp16 = inFp16 && outFp16;
    }

    if (enableDummyOpReplacement) {
        registerInterfacesWithReplacement(registry);
    }

    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "MLIRContext");

    mlir::MLIRContext ctx(registry);
    addLogging(ctx, log);

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "PassManager");

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    addLogging(pm, log);
    devConf.setup(pm);

    auto rootTiming = tm.getRootScope();

    // TODO: somehow protect non-target cases
    buildPipeline(pm, config, rootTiming, log, inAndOutFp16, floatInputPrecision);  // builds compilation pipeline
    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "prepareNetwork");

    const auto cnnNet = prepareNetwork(func, inputsInfo, outputsInfo, rootTiming);
    std::vector<vpux::PreProcessInfo> preProcInfo;

    const auto arch = getArchKind(config);

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "importNetwork");

    const auto module = importNetwork(&ctx, cnnNet, devConf, preProcInfo, rootTiming, config.get<PERF_COUNT>(),
                                      enableDummyOpReplacement, arch, log);

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "compileNetwork");

    compileNetwork(module.get(), pm, rootTiming);  // applies each pass in the pipeline

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "exportNetwork");

    const auto blob =
            exportToELF(module.get(), preProcInfo, buildOVParams(func, inputsInfo), buildOVResults(func, outputsInfo));
    std::vector<char> compiledNetwork(blob.size());
    std::copy_n(reinterpret_cast<const char*>(blob.data()), blob.size(), compiledNetwork.data());
    return std::make_shared<VPUIPRegMapped::NetworkDescription>(std::move(compiledNetwork));

    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);
}

//
// CompilerImpl::parse
//

std::shared_ptr<vpux::INetworkDescription> vpux::CompilerImpl::parse(const std::vector<char>& compiledNetwork,
                                                                     const Config&, const std::string&) {
    return std::make_shared<VPUIP::NetworkDescription>(compiledNetwork);
}

//
// CreateVPUXCompiler
//

#ifndef OPENVINO_STATIC_LIBRARY
INFERENCE_PLUGIN_API(void) CreateVPUXCompiler(std::shared_ptr<ICompiler>& obj) {
    obj = std::make_shared<CompilerImpl>();
}
#endif
