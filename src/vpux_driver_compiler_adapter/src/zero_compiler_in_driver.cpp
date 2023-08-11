//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include "zero_compiler_in_driver.h"
#include "ie_layouts.h"
#include "vpux/al/config/common.hpp"
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/IE/prefix.hpp"
#define UNUSED(x) (void)(x)

namespace vpux {
namespace driverCompilerAdapter {

namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
// TODO #-30200 : Not all Precision from IE listed in ze_graph_ext
// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
IE::Precision toIEPrecision(const ze_graph_argument_precision_t zePrecision) {
    switch (zePrecision) {
    case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
        return IE::Precision::FP32;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
        return IE::Precision::FP16;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
        return IE::Precision::U16;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
        return IE::Precision::U8;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT32:
        return IE::Precision::I32;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
        return IE::Precision::I16;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
        return IE::Precision::I8;
    case ZE_GRAPH_ARGUMENT_PRECISION_BIN:
        return IE::Precision::BIN;

    default:
        return IE::Precision::UNSPECIFIED;
    }
}

ov::element::Type_t toOVElementType(const ze_graph_metadata_type zeElementType) {
    switch (zeElementType) {
    case ZE_GRAPH_METADATA_TYPE_UNDEFINED:
        return ov::element::Type_t::undefined;
    case ZE_GRAPH_METADATA_TYPE_DYNAMIC:
        return ov::element::Type_t::dynamic;
    case ZE_GRAPH_METADATA_TYPE_BOOLEAN:
        return ov::element::Type_t::boolean;
    case ZE_GRAPH_METADATA_TYPE_BF16:
        return ov::element::Type_t::bf16;
    case ZE_GRAPH_METADATA_TYPE_F16:
        return ov::element::Type_t::f16;
    case ZE_GRAPH_METADATA_TYPE_F32:
        return ov::element::Type_t::f32;
    case ZE_GRAPH_METADATA_TYPE_F64:
        return ov::element::Type_t::f64;
    case ZE_GRAPH_METADATA_TYPE_I4:
        return ov::element::Type_t::i4;
    case ZE_GRAPH_METADATA_TYPE_I8:
        return ov::element::Type_t::i8;
    case ZE_GRAPH_METADATA_TYPE_I16:
        return ov::element::Type_t::i16;
    case ZE_GRAPH_METADATA_TYPE_I32:
        return ov::element::Type_t::i32;
    case ZE_GRAPH_METADATA_TYPE_I64:
        return ov::element::Type_t::i64;
    case ZE_GRAPH_METADATA_TYPE_U1:
        return ov::element::Type_t::u1;
    case ZE_GRAPH_METADATA_TYPE_U4:
        return ov::element::Type_t::u4;
    case ZE_GRAPH_METADATA_TYPE_U8:
        return ov::element::Type_t::u8;
    case ZE_GRAPH_METADATA_TYPE_U16:
        return ov::element::Type_t::u16;
    case ZE_GRAPH_METADATA_TYPE_U32:
        return ov::element::Type_t::u32;
    case ZE_GRAPH_METADATA_TYPE_U64:
        return ov::element::Type_t::u64;
    default:
        return ov::element::Type_t::undefined;
    }
}

// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
IE::Layout toIELayout(const ze_graph_argument_layout_t zeLayout) {
    switch (zeLayout) {
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCHW:
        return IE::Layout::NCHW;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NHWC:
        return IE::Layout::NHWC;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW:
        return IE::Layout::NCDHW;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC:
        return IE::Layout::NDHWC;

    case ZE_GRAPH_ARGUMENT_LAYOUT_OIHW:
        return IE::Layout::OIHW;

    case ZE_GRAPH_ARGUMENT_LAYOUT_C:
        return IE::Layout::C;

    case ZE_GRAPH_ARGUMENT_LAYOUT_CHW:
        return IE::Layout::CHW;

    case ZE_GRAPH_ARGUMENT_LAYOUT_HW:
        return IE::Layout::HW;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NC:
        return IE::Layout::NC;
    case ZE_GRAPH_ARGUMENT_LAYOUT_CN:
        return IE::Layout::CN;

    case ZE_GRAPH_ARGUMENT_LAYOUT_BLOCKED:
        return IE::Layout::BLOCKED;
    default:
        return IE::Layout::ANY;
    }
}

// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
size_t getDimCount(const IE::Layout layout) {
    switch (layout) {
    case IE::Layout::C:
        return 1;
    case IE::Layout::CN:
        return 2;
    case IE::Layout::HW:
        return 2;
    case IE::Layout::NC:
        return 2;
    case IE::Layout::CHW:
        return 3;
    case IE::Layout::NCHW:
        return 4;
    case IE::Layout::NHWC:
        return 4;
    case IE::Layout::NCDHW:
        return 5;
    case IE::Layout::NDHWC:
        return 5;
    default:
        // TODO #-30200 Extend to support all cases
        return 0;
    }

    return 0;
}

// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
ze_graph_argument_layout_t toZeLayout(const IE::Layout layout) {
    switch (layout) {
    case IE::Layout::NCHW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NCHW;
    case IE::Layout::NHWC:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NHWC;
    case IE::Layout::NCDHW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW;
    case IE::Layout::NDHWC:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC;

    case IE::Layout::OIHW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_OIHW;

    case IE::Layout::C:
        return ZE_GRAPH_ARGUMENT_LAYOUT_C;

    case IE::Layout::CHW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_CHW;

    case IE::Layout::HW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_HW;
    case IE::Layout::NC:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NC;
    case IE::Layout::CN:
        return ZE_GRAPH_ARGUMENT_LAYOUT_CN;

    case IE::Layout::BLOCKED:
        return ZE_GRAPH_ARGUMENT_LAYOUT_BLOCKED;
    default:
        return ZE_GRAPH_ARGUMENT_LAYOUT_ANY;
    }
}

// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
ze_graph_argument_precision_t toZePrecision(const IE::Precision precision) {
    switch (precision) {
    case IE::Precision::I8:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT8;
    case IE::Precision::U8:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT8;
    case IE::Precision::I16:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT16;
    case IE::Precision::U16:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT16;
    case IE::Precision::I32:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT32;
    case IE::Precision::FP16:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP16;
    case IE::Precision::FP32:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP32;
    case IE::Precision::BIN:
        return ZE_GRAPH_ARGUMENT_PRECISION_BIN;
    default:
        return ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN;
    }
}

template <typename TableExtension>
LevelZeroCompilerInDriver<TableExtension>::~LevelZeroCompilerInDriver() {
    if (_context) {
        auto result = zeContextDestroy(_context);
        if (ZE_RESULT_SUCCESS != result) {
            _logger.warning("zeContextDestroy failed {0:X+}", uint64_t(result));
        }
    }
    _logger.debug("LevelZeroCompilerInDriver obj destroyed");
}

using SerializedIR = std::vector<uint8_t>;
/**
 * @brief Place xml + weights in sequential memory
 * @details Format of the memory:
 */
template <typename TableExtension>
SerializedIR LevelZeroCompilerInDriver<TableExtension>::serializeIR(const std::vector<char>& xml,
                                                                    const std::vector<char>& weights) {
    // Contract between adapter and compiler in driver
    const uint32_t maxNumberOfElements = 10;
    const uint64_t maxSizeOfXML = std::numeric_limits<uint64_t>::max() / 3;
    const uint64_t maxSizeOfWeights = maxSizeOfXML * 2;

    ze_device_graph_properties_t deviceGraphProperties{};

    auto result = _graphDdiTableExt->pfnDeviceGetGraphProperties(_deviceHandle, &deviceGraphProperties);

    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerInDriver: Failed to get graph properties from compiler";
    }

    const auto compilerVersion = deviceGraphProperties.compilerVersion;

    const uint32_t numberOfInputData = 2;
    const uint64_t xmlSize = static_cast<uint64_t>(xml.size());
    const uint64_t weightsSize = static_cast<uint64_t>(weights.size());

    IE_ASSERT(numberOfInputData < maxNumberOfElements);
    if (xmlSize >= maxSizeOfXML) {
        IE_THROW() << "LevelZeroCompilerInDriver: Xml file is too big to process.";
    }
    if (weightsSize >= maxSizeOfWeights) {
        IE_THROW() << "LevelZeroCompilerInDriver: Bin file is too big to process.";
    }

    const uint64_t sizeOfSerializedIR = sizeof(compilerVersion) + sizeof(numberOfInputData) + sizeof(xmlSize) +
                                        xml.size() + sizeof(weightsSize) + weights.size();

    std::vector<uint8_t> serializedIR;
    serializedIR.resize(sizeOfSerializedIR);

    uint64_t offset = 0;
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &compilerVersion, sizeof(compilerVersion));
    offset += sizeof(compilerVersion);

    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &numberOfInputData, sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, xml.data(), xmlSize);
    offset += xmlSize;
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    ie_memcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, weights.data(), weightsSize);
    offset += weightsSize;

    IE_ASSERT(offset == sizeOfSerializedIR);

    return serializedIR;
}

/**
 * @brief Serialize input / output information to string format
 * Format:
 * --inputs_precisions="<input1Name>:<input1Precision> [<input2Name>:<input2Precision>]"
 * --inputs_layouts="<input1Name>:<input1Layout> [<input2Name>:<input2Layout>]"
 * --outputs_precisions="<output1Name>:<output1Precision>"
 * --outputs_layouts="<output1Name>:<output1Layout>"
 */
template <typename TableExtension>
std::string LevelZeroCompilerInDriver<TableExtension>::serializeIOInfo(
        const InferenceEngine::InputsDataMap& inputsInfo, const InferenceEngine::OutputsDataMap& outputsInfo) {
    const std::string inputsPrecisionsKey = "--inputs_precisions";
    const std::string inputsLayoutsKey = "--inputs_layouts";
    const std::string outputsPrecisionsKey = "--outputs_precisions";
    const std::string outputsLayoutsKey = "--outputs_layouts";

    // <option key>="<option value>"
    const std::string keyValueSeparator = "=";
    const std::string valueDelimiter = "\"";  // marks beginning and end of value

    // Format inside "<option value>"
    // <name1>:<value (precision / layout)> [<name2>:<value>]
    const std::string nameValueSeparator = ":";
    const std::string valuesSeparator = " ";

    auto serializeOptionValue = [&](auto& portsInfo, auto& precisionSS, auto& layoutSS) {
        for (auto&& port : portsInfo) {
            if (port.first != portsInfo.cbegin()->first) {
                precisionSS << valuesSeparator;
                layoutSS << valuesSeparator;
            }
            precisionSS << port.first << nameValueSeparator << port.second->getPrecision();
            layoutSS << port.first << nameValueSeparator << port.second->getLayout();
        }

        precisionSS << valueDelimiter;
        layoutSS << valueDelimiter;
    };

    std::stringstream inputsPrecisionSS;
    std::stringstream inputsLayoutSS;

    inputsPrecisionSS << inputsPrecisionsKey << keyValueSeparator << valueDelimiter;
    inputsLayoutSS << inputsLayoutsKey << keyValueSeparator << valueDelimiter;

    serializeOptionValue(inputsInfo, inputsPrecisionSS, inputsLayoutSS);

    std::stringstream outputsPrecisionSS;
    std::stringstream outputsLayoutSS;

    outputsPrecisionSS << outputsPrecisionsKey << keyValueSeparator << valueDelimiter;
    outputsLayoutSS << outputsLayoutsKey << keyValueSeparator << valueDelimiter;

    serializeOptionValue(outputsInfo, outputsPrecisionSS, outputsLayoutSS);

    // One line without spaces to avoid parsing as config option inside CID
    return inputsPrecisionSS.str() + valuesSeparator + inputsLayoutSS.str() + valuesSeparator +
           outputsPrecisionSS.str() + valuesSeparator + outputsLayoutSS.str();
}

std::string toString(const ze_graph_argument_precision_t& precision) {
    switch (precision) {
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
        return "UINT8";
    case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
        return "FP32";
    case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
        return "FP16";
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
        return "UINT16";
    case ZE_GRAPH_ARGUMENT_PRECISION_INT32:
        return "INT32";
    case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
        return "INT16";
    case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
        return "INT8";
    case ZE_GRAPH_ARGUMENT_PRECISION_BIN:
        return "BIN";
    case ZE_GRAPH_ARGUMENT_PRECISION_BF16:
        return "BF16";
    default:
        return "UNKNOWN";
    }
}

std::string toString(const ze_graph_argument_layout_t& layout) {
    switch (layout) {
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCHW:
        return "NCHW";
    case ZE_GRAPH_ARGUMENT_LAYOUT_NHWC:
        return "NHWC";
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW:
        return "NCDHW";
    case ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC:
        return "NDHWC";
    case ZE_GRAPH_ARGUMENT_LAYOUT_OIHW:
        return "OIHW";
    case ZE_GRAPH_ARGUMENT_LAYOUT_C:
        return "C";
    case ZE_GRAPH_ARGUMENT_LAYOUT_CHW:
        return "CHW";
    case ZE_GRAPH_ARGUMENT_LAYOUT_HW:
        return "HW";
    case ZE_GRAPH_ARGUMENT_LAYOUT_NC:
        return "NC";
    case ZE_GRAPH_ARGUMENT_LAYOUT_CN:
        return "CN";
    case ZE_GRAPH_ARGUMENT_LAYOUT_BLOCKED:
        return "BLOCKED";
    default:
        return "ANY";
    }
}

static std::string serializeConfig(const vpux::Config& config) {
    return "--config " + config.toString();
}

template <typename TableExtension>
INetworkDescription::Ptr LevelZeroCompilerInDriver<TableExtension>::compileIR(
        const std::string& graphName, const std::vector<char>& xml, const std::vector<char>& weights,
        const IE::InputsDataMap& inputsInfo, const IE::OutputsDataMap& outputsInfo, const vpux::Config& config) {
    _logger.setLevel(config.get<LOG_LEVEL>());
    _logger.debug("LevelZeroCompilerInDriver::compileIR");
    auto serializedIR = serializeIR(xml, weights);

    ze_graph_format_t format = ZE_GRAPH_FORMAT_NGRAPH_LITE;

    if (inputsInfo.empty() || outputsInfo.empty()) {
        THROW_IE_EXCEPTION << "Information about inputs or outputs is not provided.";
    }

    std::string buildFlags;

    buildFlags += serializeIOInfo(inputsInfo, outputsInfo);
    buildFlags += " ";
    buildFlags += serializeConfig(config);

    _logger.debug("Build flags : {0}", buildFlags);

    ze_graph_desc_t desc{ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                         nullptr,
                         format,
                         serializedIR.size(),
                         serializedIR.data(),
                         buildFlags.c_str()};

    // TODO #-30202 Store graph_handle inside NetworkDesc instead of blob. But this will require changes in zeroAPI

    // Graph handle should be used only in scope of compile / parse functions.
    ze_graph_handle_t graphHandle;

    _logger.info("Using extension version: {0}", typeid(TableExtension).name());
    auto result = _graphDdiTableExt->pfnCreate(_context, _deviceHandle, &desc, &graphHandle);

    VPUX_THROW_WHEN(result != ZE_RESULT_SUCCESS,
                    "LevelZeroCompilerInDriver: Failed to compile network. Error code: {0}. {1}", result,
                    getLatestBuildError());

    // Get blob size first
    size_t blobSize = -1;

    result = _graphDdiTableExt->pfnGetNativeBinary(graphHandle, &blobSize, nullptr);

    VPUX_THROW_WHEN(result != ZE_RESULT_SUCCESS,
                    "LevelZeroCompilerInDriver: Failed to get blob size. Error code: {0}. {1}", result,
                    getLatestBuildError());

    std::vector<char> blob(blobSize);
    // Get blob data
    result = _graphDdiTableExt->pfnGetNativeBinary(graphHandle, &blobSize, reinterpret_cast<uint8_t*>(blob.data()));

    VPUX_THROW_WHEN(result != ZE_RESULT_SUCCESS,
                    "LevelZeroCompilerInDriver: Failed to get compiled network. Error code: {0}. {1}", result,
                    getLatestBuildError());

    const auto networkMeta = getNetworkMeta(graphHandle);
    result = _graphDdiTableExt->pfnDestroy(graphHandle);

    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerInDriver: Failed to destroy graph handle. Error code: " << std::hex << result;
    }

    _logger.debug("LevelZeroCompilerInDriver::compileIR end");
    return std::make_shared<driverCompilerAdapter::NetworkDescription>(blob, graphName, networkMeta);
}

template <typename TableExtension>
std::shared_ptr<INetworkDescription> LevelZeroCompilerInDriver<TableExtension>::parseBlob(const std::string& graphName,
                                                                                          const std::vector<char>& blob,
                                                                                          const vpux::Config& config) {
    OV_ITT_TASK_CHAIN(PARSE_BLOB, itt::domains::VPUXPlugin, "LevelZeroCompilerInDriver::parseBlob", "desc");
    _logger.setLevel(config.get<LOG_LEVEL>());
    _logger.debug("LevelZeroCompilerInDriver::getNetworkMeta");
    ze_graph_handle_t graphHandle;

    if (!blob.empty()) {
        _logger.debug("Import network case");
        ze_graph_format_t format = ZE_GRAPH_FORMAT_NATIVE;
        ze_graph_desc_t desc{ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,       nullptr, format, blob.size(),
                             reinterpret_cast<const uint8_t*>(blob.data()), nullptr};

        auto result = _graphDdiTableExt->pfnCreate(_context, _deviceHandle, &desc, &graphHandle);
        OV_ITT_TASK_NEXT(PARSE_BLOB, "_graphDdiTableExt");

        if (ZE_RESULT_SUCCESS != result) {
            IE_THROW() << "LevelZeroCompilerInDriver: Failed to import blob. Error code: " << std::hex << result;
        }
    } else {
        THROW_IE_EXCEPTION << "Empty blob";
    }

    OV_ITT_TASK_NEXT(PARSE_BLOB, "getNetworkMeta");
    const auto networkMeta = getNetworkMeta(graphHandle);
    OV_ITT_TASK_NEXT(PARSE_BLOB, "NetworkDescription");

    auto result = _graphDdiTableExt->pfnDestroy(graphHandle);

    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerInDriver: Failed to destroy graph handle. Error code: " << std::hex << result;
    }

    return std::make_shared<driverCompilerAdapter::NetworkDescription>(blob, graphName, networkMeta);
}

template <typename TableExtension>
size_t LevelZeroCompilerInDriver<TableExtension>::getSupportedOpset() {
    _logger.debug("LevelZeroCompilerInDriver::getSupportedOpset");
    ze_device_graph_properties_t graphProperties;

    auto result = _graphDdiTableExt->pfnDeviceGetGraphProperties(_deviceHandle, &graphProperties);

    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerInDriver: Failed to get opset version from compiler";
    }
    const auto maxOpsetVersion = graphProperties.maxOVOpsetVersionSupported;
    _logger.info("Max supported version of opset in CiD: {0}", maxOpsetVersion);
    return maxOpsetVersion;
}

template <typename TableExtension>
template <typename T>
void LevelZeroCompilerInDriver<TableExtension>::getNetDevInOutputs(DataMap& netInputs, DataMap& devInputs,
                                                                   DataMap& netOutputs, DataMap& devOutputs,
                                                                   const T& arg) {
    IE::Precision netPrecision = toIEPrecision(arg.networkPrecision);
    IE::Layout netLayout = toIELayout(arg.networkLayout);
    IE::SizeVector netDims(arg.dims, arg.dims + getDimCount(netLayout));
    IE::TensorDesc netDataDesc(netPrecision, netDims, netLayout);

    IE::Precision dev_precision = toIEPrecision(arg.devicePrecision);
    IE::Layout devLayout = toIELayout(arg.deviceLayout);
    IE::SizeVector devDims(arg.dims, arg.dims + getDimCount(devLayout));
    IE::TensorDesc devDataDesc(dev_precision, devDims, devLayout);

    if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
        _logger.info("Found input \"{0}\"", arg.name);
        netInputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, netDataDesc));
        devInputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, devDataDesc));
    }

    if (ZE_GRAPH_ARGUMENT_TYPE_OUTPUT == arg.type) {
        _logger.info("Found output \"{0}\"", arg.name);
        netOutputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, netDataDesc));
        devOutputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, devDataDesc));
    }
}

void getOVNodes(std::vector<OVRawNode>& ovInfo, ze_graph_argument_metadata_t& metaData) {
    ov::Shape nodeShape;
    std::unordered_set<std::string> nodeTensorsNames;

    for (uint32_t id = 0; id < metaData.tensor_names_count; id++) {
        nodeTensorsNames.insert(metaData.tensor_names[id]);
    }
    for (uint32_t id = 0; id < metaData.shape_size; id++) {
        nodeShape.push_back(metaData.shape[id]);
    }
    ov::element::Type_t nodeType = toOVElementType(metaData.data_type);

    ovInfo.push_back({metaData.friendly_name, nodeType, nodeShape, std::move(nodeTensorsNames), metaData.input_name});
}

template <>
void LevelZeroCompilerInDriver<ze_graph_dditable_ext_t>::getMetaData(ze_graph_dditable_ext_t* graphDdiTableExt,
                                                                     ze_graph_handle_t graphHandle, uint32_t index,
                                                                     DataMap& netInputs, DataMap& devInputs,
                                                                     DataMap& netOutputs, DataMap& devOutputs,
                                                                     std::vector<OVRawNode>& ovResults,
                                                                     std::vector<OVRawNode>& ovParameters) {
    ze_graph_argument_properties_t arg;
    auto result = graphDdiTableExt->pfnGetArgumentProperties(graphHandle, index, &arg);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerInDriver: Failed to get information about inputs/outputs.";
    }

    getNetDevInOutputs(netInputs, devInputs, netOutputs, devOutputs, arg);
    _logger.warning("ovResults is unsupported, thus stays unchanged.");
    _logger.warning("ovParameters is unsupported, thus stays unchanged.");
    UNUSED(ovResults);
    UNUSED(ovParameters);
}

template <typename TableExtension>
void LevelZeroCompilerInDriver<TableExtension>::getMetaData(TableExtension* graphDdiTableExt,
                                                            ze_graph_handle_t graphHandle, uint32_t index,
                                                            DataMap& netInputs, DataMap& devInputs, DataMap& netOutputs,
                                                            DataMap& devOutputs, std::vector<OVRawNode>& ovResults,
                                                            std::vector<OVRawNode>& ovParameters) {
    ze_graph_argument_properties_2_t arg;
    auto result = graphDdiTableExt->pfnGetArgumentProperties2(graphHandle, index, &arg);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerInDriver: Failed to get information about inputs/outputs.";
    }

    getNetDevInOutputs(netInputs, devInputs, netOutputs, devOutputs, arg);

    // The I/O data corresponding to the states of the model is not found within the OpenVINO 2.0 attributes contained
    // by the compiled model, thus we should not query them
    if (!isStateInputName(arg.name) && !isStateOutputName(arg.name)) {
        ze_graph_argument_metadata_t metaData;
        result = graphDdiTableExt->pfnGraphGetArgumentMetadata(graphHandle, index, &metaData);
        if (ZE_RESULT_SUCCESS != result) {
            IE_THROW() << "LevelZeroCompilerInDriver: Failed to get information about inputs/outputs.";
        }

        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
            getOVNodes(ovParameters, metaData);
        }

        if (ZE_GRAPH_ARGUMENT_TYPE_OUTPUT == arg.type) {
            getOVNodes(ovResults, metaData);
        }
    }
}

template <typename TableExtension>
NetworkMeta LevelZeroCompilerInDriver<TableExtension>::getNetworkMeta(ze_graph_handle_t graphHandle) {
    ze_graph_properties_t graphProperties{};

    auto result = _graphDdiTableExt->pfnGetProperties(graphHandle, &graphProperties);

    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerInDriver: Failed to get information about graph.";
    }

    DataMap netInputs;
    DataMap devInputs;

    DataMap netOutputs;
    DataMap devOutputs;

    std::vector<OVRawNode> ovResults;
    std::vector<OVRawNode> ovParameters;
    for (uint32_t index = 0; index < graphProperties.numGraphArgs; ++index) {
        getMetaData(_graphDdiTableExt, graphHandle, index, netInputs, devInputs, netOutputs, devOutputs, ovResults,
                    ovParameters);
    }
    // TODO: support this information in CiD [track: E#33479]
    int numStreams = 1;
    return NetworkMeta{std::move(netInputs),
                       std::move(netOutputs),
                       std::move(devInputs),
                       std::move(devOutputs),
                       std::move(ovResults),
                       std::move(ovParameters),
                       numStreams};
}

template <typename TableExtension>
template <typename T, typename std::enable_if_t<!NotSupportLogHandle(T), bool>>
std::string LevelZeroCompilerInDriver<TableExtension>::getLatestBuildError() {
    _logger.debug("LevelZeroCompilerInDriver::getLatestBuildError()");

    // Get log size
    uint32_t size = 0;
    // Null graph handle to get erro log
    auto result = _graphDdiTableExt->pfnBuildLogGetString(nullptr, &size, nullptr);
    if (ZE_RESULT_SUCCESS != result) {
        // The failure will not break normal execution, only warning here
        _logger.warning("LevelZeroCompilerInDriver: Failed to get size of latest error log!");
        return "";
    }

    if (size <= 0) {
        // The failure will not break normal execution, only warning here
        _logger.warning("No error log stored in driver when error detected, may not be compiler issue!");
        return "";
    }

    // Get log content
    std::string logContent{};
    logContent.resize(size);
    result = _graphDdiTableExt->pfnBuildLogGetString(nullptr, &size, const_cast<char*>(logContent.data()));
    if (ZE_RESULT_SUCCESS != result) {
        // The failure will not break normal execution, only warning here
        _logger.warning("LevelZeroCompilerInDriver: Failed to get content of latest error log!");
        return "";
    }
    return logContent;
}

// Explicit template instantiations to avoid linker errors
template class LevelZeroCompilerInDriver<ze_graph_dditable_ext_t>;
template class LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_1_t>;
template class LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_4_t>;

}  // namespace driverCompilerAdapter
}  // namespace vpux
