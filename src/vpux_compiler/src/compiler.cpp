//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/compiler/compiler.hpp"

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/compiler.hpp"

#include "vpux/compiler/VPU30XX/pipeline_strategy.hpp"
#include "vpux/compiler/VPU30XX/pipelines.hpp"
#include "vpux/compiler/VPU37XX/pipeline_strategy.hpp"
#include "vpux/compiler/VPU37XX/pipelines.hpp"
#include "vpux/compiler/dialect/ELF/export.hpp"
#include "vpux/compiler/dialect/EMU/graph-schema/export.hpp"
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/dialect/VPU/attributes.hpp"
#include "vpux/compiler/dialect/VPUIP/graph-schema/export.hpp"
#include "vpux/compiler/dialect/VPUIP/network_description.hpp"
#include "vpux/compiler/dialect/VPUMI37XX/network_description.hpp"
#include "vpux/compiler/frontend/IE.hpp"
#include "vpux/compiler/init.hpp"
#include "vpux/compiler/interfaces_registry.hpp"
#include "vpux/compiler/options_mapper.hpp"
#include "vpux/compiler/pipelines.hpp"
#include "vpux/compiler/utils/dot_printer.hpp"
#include "vpux/compiler/utils/logging.hpp"

#include "vpux/utils/IE/data_attributes_check.hpp"
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/core/error.hpp"
#include "vpux/utils/core/optional.hpp"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/Timing.h>

#include <llvm/Support/Regex.h>
#include <llvm/Support/raw_ostream.h>

#include <openvino/core/preprocess/pre_post_process.hpp>

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
// createPipelineStrategy
//

std::unique_ptr<IPipelineStrategy> createPipelineStrategy(VPU::ArchKind arch) {
    switch (arch) {
    case VPU::ArchKind::VPUX30XX:
        return std::make_unique<PipelineStrategy30XX>();
    case VPU::ArchKind::VPUX37XX:
        return std::make_unique<PipelineStrategy37XX>();
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
}

//
// DeveloperConfig
//

class DeveloperConfig final {
public:
    explicit DeveloperConfig(Logger log);
    DeveloperConfig(const DeveloperConfig& other) = delete;
    DeveloperConfig& operator=(const DeveloperConfig& other) = delete;
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
    parseEnv("IE_NPU_CRASH_REPRODUCER_FILE", _crashReproducerFile);
    parseEnv("IE_NPU_GEN_LOCAL_REPRODUCER", _localReproducer);

    parseEnv("IE_NPU_IR_PRINTING_FILTER", _irPrintingFilter);
    parseEnv("IE_NPU_IR_PRINTING_FILE", _irPrintingFile);
    parseEnv("IE_NPU_IR_PRINTING_ORDER", _irPrintingOrderStr);
    parseEnv("IE_NPU_PRINT_FULL_IR", _printFullIR);
    parseEnv("IE_NPU_PRINT_FULL_CONSTANT", _printFullConstant);
    parseEnv("IE_NPU_PRINT_DEBUG_INFO", _printDebugInfo);

    parseEnv("IE_NPU_PRINT_DOT", _printDotOptions);
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
                       "case-sensitive.\nExample: IE_NPU_IR_PRINTING_ORDER=Before",
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

/**
 * @brief Adds names to the tensors corresponding to the model's I/O nodes.
 * @details In the IRv10 model case, the names of the I/O tensors seem to be missing. These names
 * are requested by the "ov::PrePostProcessor" object when trying to retrieve an input/output by name,
 * thus we should add these names and use the corresponding nodes' names as the their values.
 * @param model The model representation corresponding to the 2.0 OpenVINO API version.
 */
void addIOTensorNames(const std::shared_ptr<ov::Model>& model) {
    const ov::ParameterVector& parameters = model->get_parameters();
    const ov::ResultVector& results = model->get_results();

    for (const auto& parameter : parameters) {
        const std::string& nodeName = parameter->get_friendly_name();
        ov::descriptor::Tensor& outputTensor = parameter->get_output_tensor(0);
        outputTensor.add_names({nodeName});
    }

    for (const auto& result : results) {
        const std::string& nodeName = ov::op::util::get_ie_output_name(result->input_value(0));
        ov::descriptor::Tensor& outputTensor = result->get_output_tensor(0);
        outputTensor.add_names({nodeName});
    }
}

/**
 * @brief Extracts and converts the relevant I/O metadata from the legacy model representation.
 * @details The I/O information which should be passed on to the "ov::Model" object consists
 * of the precision, layout and preprocessing data.
 *
 * This function aims to extract the previously mentioned metadata and convert it into a representation
 * which is easy to apply on the structure corresponding to the 2.0 OpenVINO API.
 * @param network The model representation used by the 1.0 API version.
 * @param inputPrecision Empty map which shall be populated with the extracted input precision information.
 * @param outputPrecision Similar to the previous parameter but applied on the outputs.
 * @param inputDimensionIndices Empty map which shall be populated with the dimension indices of
 * the inputs (e.g. "NHWC" shall be [0, 2, 3, 1]). This representation was chosen in order to offer
 * a higher degree of flexibility in expressing the shape of the data.
 * @param outputDimensionIndices Similar to the previous parameter but applied on the outputs.
 */
void extractCNNNetworkData(const InputsDataMap& inputMetadata, const OutputsDataMap& outputMetadata,
                           std::map<std::string, ov::element::Type>& inputPrecisions,
                           std::map<std::string, ov::element::Type>& outputPrecisions,
                           std::map<std::string, std::vector<uint64_t>>& inputDimensionIndices,
                           std::map<std::string, std::vector<uint64_t>>& outputDimensionIndices) {
    for (auto&& layer : inputMetadata) {
        const InputInfo::Ptr& inputInfo = layer.second;
        const std::string& inputName = inputInfo->name();
        const Precision& iePrecision = inputInfo->getPrecision();
        const ov::element::Type& ovPrecision = InferenceEngine::details::convertPrecision(iePrecision);
        const std::vector<uint64_t>& dimensionIndices = inputInfo->getTensorDesc().getBlockingDesc().getOrder();

        inputPrecisions.insert({inputName, ovPrecision});
        inputDimensionIndices.insert({inputName, dimensionIndices});
    }

    for (auto&& layer : outputMetadata) {
        const DataPtr& outputInfo = layer.second;
        const std::string& inputName = outputInfo->getName();
        const Precision& iePrecision = outputInfo->getPrecision();
        const ov::element::Type& ovPrecision = InferenceEngine::details::convertPrecision(iePrecision);
        const std::vector<uint64_t>& dimensionIndices = outputInfo->getTensorDesc().getBlockingDesc().getOrder();

        outputPrecisions.insert({inputName, ovPrecision});
        outputDimensionIndices.insert({inputName, dimensionIndices});
    }
}

/**
 * @brief Converts a list of dimension indices to the string representation required for applying
 * the "ov::Layout" class' constructor.
 * @details Output sample: "[0,2,1,3,4,..,10,11,..]". In this example the input is just a vector
 * containing the corresponding values.
 * @param dimensionIndices The vector for which the string representation shall be generated.
 * @return The string representation corresponding to the input vector.
 */
std::string vectorToStringRepresentation(const std::vector<uint64_t>& dimensionIndices) {
    std::stringstream dimensionIndicesRepresentation;
    dimensionIndicesRepresentation << '[';

    for (auto iterator = dimensionIndices.begin(); iterator != dimensionIndices.end(); ++iterator) {
        if (iterator != dimensionIndices.begin()) {
            dimensionIndicesRepresentation << ',';
        }
        dimensionIndicesRepresentation << *iterator;
    }

    dimensionIndicesRepresentation << ']';
    return dimensionIndicesRepresentation.str();
}

/**
 * @brief Applies the I/O metadata extracted from the "IE:CNNNetwork" object on
 * the "ov::Model" one.
 * @details The "ov::PrePostProcessor" tool is used for applying the changes described
 * by the extracted metadata (the precision, layout and preprocessing information).
 * @param model The model representation corresponding to tge 2.0 API version.
 * @param inputPrecisions The precision which shall be applied on the input metadata.
 * @param outputPrecisions The precision which shall be applied on the output metadata.
 * @param inputDimensionIndices The order of the input dimension indices.
 * @param outputDimensionIndices The order of the output dimension indices.
 */
void applyOnOVModel(std::shared_ptr<ov::Model>& model, const std::map<std::string, ov::element::Type>& inputPrecisions,
                    const std::map<std::string, ov::element::Type>& outputPrecisions,
                    const std::map<std::string, std::vector<uint64_t>>& inputDimensionIndices,
                    const std::map<std::string, std::vector<uint64_t>>& outputDimensionIndices) {
    auto preprocessor = ov::preprocess::PrePostProcessor(model);

    for (const auto& inputPrecisionEntry : inputPrecisions) {
        const std::string& inputName = inputPrecisionEntry.first;
        const ov::element::Type& precision = inputPrecisionEntry.second;

        // The "ov::PrePostProcessor" object requires both the modified and default layouts
        // (i.e. the transposed and sorted dimension indices) in order to modify the model's
        // layout by adding a transposition layer
        const std::vector<uint64_t>& dimensionIndices = inputDimensionIndices.at(inputName);
        std::vector<uint64_t> sortedDimensionIndices(dimensionIndices.size());
        std::iota(std::begin(sortedDimensionIndices), std::end(sortedDimensionIndices), 0);

        const std::string& tensorLayoutRepresentation = vectorToStringRepresentation(dimensionIndices);
        const std::string& modelLayoutRepresentation = vectorToStringRepresentation(sortedDimensionIndices);
        const ov::Layout tensorLayout(tensorLayoutRepresentation);
        const ov::Layout modelLayout(modelLayoutRepresentation);

        ov::preprocess::InputInfo& inputInfo = preprocessor.input(inputName);
        inputInfo.tensor().set_layout(tensorLayout);
        inputInfo.model().set_layout(modelLayout);
        inputInfo.tensor().set_element_type(precision);
    }

    for (const auto& outputPrecisionsEntry : outputPrecisions) {
        const std::string& outputName = outputPrecisionsEntry.first;
        const ov::element::Type& precision = outputPrecisionsEntry.second;

        const std::vector<uint64_t>& dimensionIndices = outputDimensionIndices.at(outputName);
        std::vector<uint64_t> sortedDimensionIndices(dimensionIndices.size());
        std::iota(std::begin(sortedDimensionIndices), std::end(sortedDimensionIndices), 0);

        const std::string& tensorLayoutRepresentation = vectorToStringRepresentation(dimensionIndices);
        const std::string& modelLayoutRepresentation = vectorToStringRepresentation(sortedDimensionIndices);
        const ov::Layout tensorLayout(tensorLayoutRepresentation);
        const ov::Layout modelLayout(modelLayoutRepresentation);

        ov::preprocess::OutputInfo& outputInfo = preprocessor.output(outputName);
        outputInfo.tensor().set_layout(tensorLayout);
        outputInfo.model().set_layout(modelLayout);
        outputInfo.tensor().set_element_type(precision);
    }

    model = preprocessor.build();
}

/**
 * @brief Sets the model's I/O metadata to the right values by copying their correspondent
 * fields from the "CNNNetwork" structure.
 * @details When using the OpenVINO 1.0 API, there are some missalignments between the
 * "CNNNetwork" and "ov::Model" objects in terms of the I/O metadata. As a consequence
 * of the Plugin API 2.0 refactoring, only the "ov::Model" object shall be moved forward
 * in the compilation flow, and for this reason the 2.0 API model object should hold the "real" values.
 *
 * In this case, the "CNNNetwork" object is holding the right metadata values, thus this function aims
 * to convert these values to the newer API and apply them on the "ov::Model" object.
 * @param model The model representation corresponding to the 2.0 API. Inside its runtime information attributes of type
 * "ov::Any" can be found the legacy I/O metadata which shall be applied on the 2.0 API version structures.
 */
void moveCNNNetworkDataToOVModel(std::shared_ptr<ov::Model>& model) {
    std::map<std::string, ov::element::Type> inputPrecisions, outputPrecisions;
    std::map<std::string, std::vector<uint64_t>> inputDimensionIndices, outputDimensionIndices;

    ov::RTMap& runtimeInfoMap = model->get_rt_info();
    const auto& inputMetadataMatch = runtimeInfoMap.find("input_metadata");
    const auto& outputMetadataMatch = runtimeInfoMap.find("output_metadata");
    if (inputMetadataMatch == runtimeInfoMap.end() || outputMetadataMatch == runtimeInfoMap.end()) {
        THROW_IE_EXCEPTION << "The I/O metadata within the model is missing.";
    }

    const auto& inputMetadata = inputMetadataMatch->second.as<InferenceEngine::InputsDataMap>();
    const auto& outputMetadata = outputMetadataMatch->second.as<InferenceEngine::OutputsDataMap>();
    if (inputMetadata.empty() || outputMetadata.empty()) {
        THROW_IE_EXCEPTION << "Empty I/O metadata";
    }

    addIOTensorNames(model);
    extractCNNNetworkData(inputMetadata, outputMetadata, inputPrecisions, outputPrecisions, inputDimensionIndices,
                          outputDimensionIndices);
    applyOnOVModel(model, inputPrecisions, outputPrecisions, inputDimensionIndices, outputDimensionIndices);
}

}  // namespace

//
// CompilerImpl::query
//

ov::SupportedOpsMap vpux::CompilerImpl::query(const std::shared_ptr<const ov::Model>& model, const Config& config) {
    Logger log("vpux-compiler", config.get<LOG_LEVEL>());
    log.setName("vpux::CompilerImpl::query");

    ov::SupportedOpsMap result;

    const std::string plugin_name = DEVICE_NAME;
    const auto arch = getArchKind(config);

    DeveloperConfig devConf(log);
    mlir::DefaultTimingManager tm;
    devConf.setup(tm);
    auto rootTiming = tm.getRootScope();

    log.trace("Get supported nodes.");
    auto supportedNodes = InferenceEngine::GetSupportedNodes(
            model,
            [&](const std::shared_ptr<ov::Model>& model) {
                log.trace("Run common nGraph passes.");
                IE::NGraphPasses::runNGraphPasses(model, rootTiming, arch);
            },
            [&](const std::shared_ptr<ngraph::Node>& op) {
                log.trace("Get supported operations list.");
                return IE::NGraphImporter::isOpSupported(op);
            });

    for (auto&& layerName : supportedNodes) {
        result.emplace(layerName, plugin_name);
    }

    return result;
}

//
// CompilerImpl::compile
//

namespace {

auto importNetwork(mlir::MLIRContext* ctx, const std::shared_ptr<ov::Model>& model, const DeveloperConfig& devConf,
                   mlir::TimingScope& rootTiming, bool enableProfiling, bool stubLayers, vpux::VPU::ArchKind arch,
                   Logger log) {
    auto importTiming = rootTiming.nest("Import network");
    return IE::importNetwork(ctx, model, devConf.useSharedConstants(), importTiming, enableProfiling, stubLayers, arch,
                             log.nest());
}

void compileNetwork(mlir::ModuleOp module, mlir::PassManager& pm, mlir::TimingScope& rootTiming) {
    auto compileTiming = rootTiming.nest("Compile network");
    pm.enableTiming(compileTiming);
    VPUX_THROW_UNLESS(mlir::succeeded(pm.run(module)), "Compilation failed");
}

auto exportToBlob(mlir::ModuleOp module, mlir::TimingScope& rootTiming,
                  const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                  const std::vector<std::shared_ptr<const ov::Node>>& results, Logger log, const Config& config) {
    auto exportTiming = rootTiming.nest("Export to blob");
    switch (config.get<PLATFORM>()) {
    case InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR:
        return EMU::exportToBlob(module, exportTiming, parameters, results, log);
    default:
        return VPUIP::exportToBlob(module, exportTiming, parameters, results, log);
    }
}

auto exportToELF(mlir::ModuleOp module, const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                 const std::vector<std::shared_ptr<const ov::Node>>& results) {
    const auto arch = VPU::getArch(module);
    switch (arch) {
    case VPU::ArchKind::VPUX37XX:
        return vpux::ELF::exportToELF(module, parameters, results);
    default:
        VPUX_THROW("Unsupported arch kind: {0}", arch);
    }
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

std::vector<std::shared_ptr<const ov::Node>> buildOVParams(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::shared_ptr<const ov::Node>> constParams;
    VPUX_THROW_WHEN(model == nullptr, "Null OV model");

    // Here we decide whether we need to add operation_names as tensor names for
    // getInputs / getOutputs. Since these functions are designed to be used in new API only
    // always need to add operation names for IR v10
    const auto addOpNames = isIR10(*model);

    for (const auto& param : model->get_parameters()) {
        auto newParam = ov::as_type_ptr<ov::op::v0::Parameter>(param->copy_with_new_inputs({}));
        newParam->set_friendly_name(param->get_friendly_name());
        if (addOpNames) {
            newParam->output(0).get_tensor().add_names({newParam->get_friendly_name()});
        }
        newParam->validate_and_infer_types();
        constParams.emplace_back(newParam);
    }

    return constParams;
}

std::vector<std::shared_ptr<const ov::Node>> buildOVResults(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::shared_ptr<const ov::Node>> constResults;
    VPUX_THROW_WHEN(model == nullptr, "Null OV model");

    // Here we decide whether we need to add operation_names as tensor names for
    // getInputs / getOutputs. Since these functions are designed to be used in new API only
    // always need to add operation names for IR v10
    const auto addOpNames = isIR10(*model);

    for (const auto& result : model->get_results()) {
        auto fakeParam = std::make_shared<ov::op::v0::Parameter>(result->get_output_element_type(0),
                                                                 result->get_output_partial_shape(0));
        const std::string paramName = ov::op::util::create_ie_output_name(result->input_value(0));
        fakeParam->set_friendly_name(paramName);
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

std::shared_ptr<INetworkDescription> exportNetwork(mlir::ModuleOp module, mlir::TimingScope& rootTiming, Logger log,
                                                   const std::shared_ptr<ov::Model>& model,
                                                   const Config& configuration) {
    const auto parameters = buildOVParams(model);
    const auto results = buildOVResults(model);

    if (isELFEnabled(configuration)) {
        const auto blob = exportToELF(module, parameters, results);
        std::vector<char> compiledNetwork(blob.size());
        std::copy_n(reinterpret_cast<const char*>(blob.data()), blob.size(), compiledNetwork.data());
        return std::make_shared<VPUMI37XX::NetworkDescription>(std::move(compiledNetwork));
    } else {
        const auto blob = exportToBlob(module, rootTiming, parameters, results, log, configuration);
        auto finalTiming = rootTiming.nest("Wrap into NetworkDescription");
        std::vector<char> compiledNetwork(blob.size());
        std::copy_n(reinterpret_cast<const char*>(blob.data()), blob.size(), compiledNetwork.data());

        return std::make_shared<VPUIP::NetworkDescription>(std::move(compiledNetwork));
    }
}

template <typename Options>
bool getDummyOpReplacement(const Config& config) {
    const auto options = Options::createFromString(config.get<COMPILATION_MODE_PARAMS>());
    VPUX_THROW_UNLESS(options != nullptr, "failed to parse COMPILATION_MODE_PARAMS");
    return options->enableDummyOpReplacement;
}

template <typename ReferenceSWOptions, typename ReferenceHWOptions, typename DefaultHWOptions>
bool getDummyOpReplacement(const Config& config) {
    const auto compilationMode = getCompilationMode(config);
    if (compilationMode == VPU::CompilationMode::ReferenceSW) {
        return getDummyOpReplacement<ReferenceSWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::ReferenceHW) {
        return getDummyOpReplacement<ReferenceHWOptions>(config);
    } else if (compilationMode == VPU::CompilationMode::DefaultHW) {
        return getDummyOpReplacement<DefaultHWOptions>(config);
    } else {
        VPUX_THROW("Unsupported compilation mode: {0}", compilationMode);
    }
}

bool getDummyOpReplacement(const Config& config) {
    const auto arch = getArchKind(config);
    if (arch == VPU::ArchKind::VPUX30XX) {
        return getDummyOpReplacement<ReferenceSWOptions30XX, ReferenceHWOptions30XX, DefaultHWOptions30XX>(config);
    } else if (arch == VPU::ArchKind::VPUX37XX) {
        return getDummyOpReplacement<ReferenceSWOptions37XX, ReferenceHWOptions37XX, DefaultHWOptions37XX>(config);
    } else {
        VPUX_THROW("Unsupported device typee: {0}", arch);
    }
}

}  // namespace

std::shared_ptr<INetworkDescription> vpux::CompilerImpl::compile(std::shared_ptr<ov::Model>& model, const std::string&,
                                                                 const Config& config) {
    OV_ITT_SCOPED_TASK(itt::domains::VPUXPlugin, "CompilerImpl::compile");
    Logger log("vpux-compiler", config.get<LOG_LEVEL>());

    DeveloperConfig devConf(log);

    mlir::DefaultTimingManager tm;
    devConf.setup(tm);

    const auto arch = getArchKind(config);

    mlir::DialectRegistry registry;
    registerDialects(registry);

    // TODO: needs refactoring. Ticket: E#50937
    // Dummy op interfaces will end up being deleted if we properly refactor this dummy op feature
    bool enableDummyOpReplacement = getDummyOpReplacement(config);
    registerCommonInterfaces(registry, enableDummyOpReplacement);

    auto interfacesRegistry = createInterfacesRegistry(arch);
    interfacesRegistry->registerInterfaces(registry);

    bool isNewAPI = false;
    ov::RTMap& runtimeInfoMap = model->get_rt_info();
    if (const auto& isNewAPIMatch = runtimeInfoMap.find("is_new_api"); isNewAPIMatch != runtimeInfoMap.end()) {
        isNewAPI = isNewAPIMatch->second.as<bool>();
    }

    // "Unifying" the metadata information found within the "ie::CNNNetwork" and "ov::Model" structures is redundant in
    // the case when the 2.0 OpenVINO API and an IRv11 model is used
    if (!isNewAPI || isIR10(*model)) {
        moveCNNNetworkDataToOVModel(model);
    }

    OV_ITT_TASK_CHAIN(COMPILER_IMPLEMENTATION, itt::domains::VPUXPlugin, "CompilerImpl::compile", "MLIRContext");

    mlir::MLIRContext ctx(registry);
    addLogging(ctx, log);
    auto rootTiming = tm.getRootScope();

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "importNetwork");

    const auto module = importNetwork(&ctx, model, devConf, rootTiming, config.get<PERF_COUNT>(),
                                      enableDummyOpReplacement, arch, log);

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "PassManager");

    mlir::PassManager pm(&ctx, mlir::OpPassManager::Nesting::Implicit);
    addLogging(pm, log);
    devConf.setup(pm);

    auto pipelineFactory = createPipelineStrategy(arch);

    // TODO: somehow protect non-target cases
    pipelineFactory->buildPipeline(pm, config, rootTiming, log);

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "compileNetwork");

    compileNetwork(module.get(), pm, rootTiming);  // applies each pass in the pipeline

    OV_ITT_TASK_NEXT(COMPILER_IMPLEMENTATION, "exportNetwork");
    const std::shared_ptr<INetworkDescription>& networkDescription =
            exportNetwork(module.get(), rootTiming, log, model, config);
    OV_ITT_TASK_SKIP(COMPILER_IMPLEMENTATION);

    return networkDescription;
}

//
// CompilerImpl::parse
//

std::shared_ptr<vpux::INetworkDescription> vpux::CompilerImpl::parse(const std::vector<char>& compiledNetwork,
                                                                     const Config& config, const std::string&) {
    if (isELFEnabled(config)) {
        return std::make_shared<VPUMI37XX::NetworkDescription>(compiledNetwork);
    } else {
        return std::make_shared<VPUIP::NetworkDescription>(compiledNetwork);
    }
}

//
// CreateVPUXCompiler
//

#ifndef OPENVINO_STATIC_LIBRARY
INFERENCE_PLUGIN_API(void) CreateVPUXCompiler(std::shared_ptr<ICompiler>& obj) {
    obj = std::make_shared<CompilerImpl>();
}
#endif

bool vpux::isELFEnabled(const Config& configuration) {
#ifdef __unix__
    const auto isUnix = true;
#else
    const auto isUnix = false;
#endif
    const auto isVPUX37XX = getArchKind(configuration) == vpux::VPU::ArchKind::VPUX37XX;
    const auto optionValue = configuration.get<USE_ELF_COMPILER_BACKEND>();
    using InferenceEngine::VPUXConfigParams::ElfCompilerBackend;

    return optionValue == ElfCompilerBackend::YES || (optionValue == ElfCompilerBackend::AUTO && isUnix && isVPUX37XX);
}
