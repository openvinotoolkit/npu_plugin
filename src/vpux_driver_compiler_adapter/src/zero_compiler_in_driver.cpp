//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "zero_compiler_in_driver.h"
#include <regex>
#include "ie_layouts.h"
#include "vpux/al/config/common.hpp"
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/IE/prefix.hpp"

#define UNUSED(x) (void)(x)

namespace vpux {
namespace driverCompilerAdapter {

namespace ie = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
// TODO #-30200 : Not all Precision from IE listed in ze_graph_ext
// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
ie::Precision toIEPrecision(const ze_graph_argument_precision_t zePrecision) {
    switch (zePrecision) {
    case ZE_GRAPH_ARGUMENT_PRECISION_INT4:
        return ie::Precision::I4;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT4:
        return ie::Precision::U4;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
        return ie::Precision::I8;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
        return ie::Precision::U8;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
        return ie::Precision::I16;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
        return ie::Precision::U16;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT32:
        return ie::Precision::I32;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT32:
        return ie::Precision::U32;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT64:
        return ie::Precision::I64;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT64:
        return ie::Precision::U64;
    case ZE_GRAPH_ARGUMENT_PRECISION_BF16:
        return ie::Precision::BF16;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
        return ie::Precision::FP16;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
        return ie::Precision::FP32;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP64:
        return ie::Precision::FP64;
    case ZE_GRAPH_ARGUMENT_PRECISION_BIN:
        return ie::Precision::BIN;
    default:
        return ie::Precision::UNSPECIFIED;
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

ov::element::Type_t toOVElementType(const ze_graph_argument_precision_t zeElementType) {
    switch (zeElementType) {
    case ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN:
        return ov::element::Type_t::undefined;
    case ZE_GRAPH_ARGUMENT_PRECISION_DYNAMIC:
        return ov::element::Type_t::dynamic;
    case ZE_GRAPH_ARGUMENT_PRECISION_BOOLEAN:
        return ov::element::Type_t::boolean;
    case ZE_GRAPH_ARGUMENT_PRECISION_BF16:
        return ov::element::Type_t::bf16;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP16:
        return ov::element::Type_t::f16;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP32:
        return ov::element::Type_t::f32;
    case ZE_GRAPH_ARGUMENT_PRECISION_FP64:
        return ov::element::Type_t::f64;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT4:
        return ov::element::Type_t::i4;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT8:
        return ov::element::Type_t::i8;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT16:
        return ov::element::Type_t::i16;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT32:
        return ov::element::Type_t::i32;
    case ZE_GRAPH_ARGUMENT_PRECISION_INT64:
        return ov::element::Type_t::i64;
    case ZE_GRAPH_ARGUMENT_PRECISION_BIN:
        return ov::element::Type_t::u1;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT4:
        return ov::element::Type_t::u4;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT8:
        return ov::element::Type_t::u8;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT16:
        return ov::element::Type_t::u16;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT32:
        return ov::element::Type_t::u32;
    case ZE_GRAPH_ARGUMENT_PRECISION_UINT64:
        return ov::element::Type_t::u64;
    default:
        return ov::element::Type_t::undefined;
    }
}

// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
ie::Layout toIELayout(const ze_graph_argument_layout_t zeLayout) {
    switch (zeLayout) {
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCHW:
        return ie::Layout::NCHW;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NHWC:
        return ie::Layout::NHWC;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW:
        return ie::Layout::NCDHW;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC:
        return ie::Layout::NDHWC;

    case ZE_GRAPH_ARGUMENT_LAYOUT_OIHW:
        return ie::Layout::OIHW;

    case ZE_GRAPH_ARGUMENT_LAYOUT_C:
        return ie::Layout::C;

    case ZE_GRAPH_ARGUMENT_LAYOUT_CHW:
        return ie::Layout::CHW;

    case ZE_GRAPH_ARGUMENT_LAYOUT_HW:
        return ie::Layout::HW;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NC:
        return ie::Layout::NC;
    case ZE_GRAPH_ARGUMENT_LAYOUT_CN:
        return ie::Layout::CN;

    case ZE_GRAPH_ARGUMENT_LAYOUT_BLOCKED:
        return ie::Layout::BLOCKED;
    default:
        return ie::Layout::ANY;
    }
}

// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
size_t getDimCount(const ie::Layout layout) {
    switch (layout) {
    case ie::Layout::C:
        return 1;
    case ie::Layout::CN:
        return 2;
    case ie::Layout::HW:
        return 2;
    case ie::Layout::NC:
        return 2;
    case ie::Layout::CHW:
        return 3;
    case ie::Layout::NCHW:
        return 4;
    case ie::Layout::NHWC:
        return 4;
    case ie::Layout::NCDHW:
        return 5;
    case ie::Layout::NDHWC:
        return 5;
    default:
        // TODO #-30200 Extend to support all cases
        return 0;
    }

    return 0;
}

// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
ze_graph_argument_layout_t toZeLayout(const ie::Layout layout) {
    switch (layout) {
    case ie::Layout::NCHW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NCHW;
    case ie::Layout::NHWC:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NHWC;
    case ie::Layout::NCDHW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW;
    case ie::Layout::NDHWC:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC;

    case ie::Layout::OIHW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_OIHW;

    case ie::Layout::C:
        return ZE_GRAPH_ARGUMENT_LAYOUT_C;

    case ie::Layout::CHW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_CHW;

    case ie::Layout::HW:
        return ZE_GRAPH_ARGUMENT_LAYOUT_HW;
    case ie::Layout::NC:
        return ZE_GRAPH_ARGUMENT_LAYOUT_NC;
    case ie::Layout::CN:
        return ZE_GRAPH_ARGUMENT_LAYOUT_CN;

    case ie::Layout::BLOCKED:
        return ZE_GRAPH_ARGUMENT_LAYOUT_BLOCKED;
    default:
        return ZE_GRAPH_ARGUMENT_LAYOUT_ANY;
    }
}

// TODO #-30406 : Remove helpers-converters duplications between driver compiler adapter and zero backend
ze_graph_argument_precision_t toZePrecision(const ie::Precision precision) {
    switch (precision) {
    case ie::Precision::I4:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT4;
    case ie::Precision::U4:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT4;
    case ie::Precision::I8:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT8;
    case ie::Precision::U8:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT8;
    case ie::Precision::I16:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT16;
    case ie::Precision::U16:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT16;
    case ie::Precision::I32:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT32;
    case ie::Precision::U32:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT32;
    case ie::Precision::I64:
        return ZE_GRAPH_ARGUMENT_PRECISION_INT64;
    case ie::Precision::U64:
        return ZE_GRAPH_ARGUMENT_PRECISION_UINT64;
    case ie::Precision::BF16:
        return ZE_GRAPH_ARGUMENT_PRECISION_BF16;
    case ie::Precision::FP16:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP16;
    case ie::Precision::FP32:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP32;
    case ie::Precision::FP64:
        return ZE_GRAPH_ARGUMENT_PRECISION_FP64;
    case ie::Precision::BIN:
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
                                                                    const std::vector<char>& weights,
                                                                    ze_graph_compiler_version_info_t& compilerVersion) {
    // Contract between adapter and compiler in driver
    const uint32_t maxNumberOfElements = 10;
    const uint64_t maxSizeOfXML = std::numeric_limits<uint64_t>::max() / 3;
    const uint64_t maxSizeOfWeights = maxSizeOfXML * 2;

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

static std::string serializeConfig(const vpux::Config& config, ze_graph_compiler_version_info_t& compilerVersion) {
    std::string content = config.toString();
    // From 5.0.0, driver compiler start to use NPU_ prefix, the old version uses VPU_ prefix
    if (compilerVersion.major < 5) {
        std::regex reg("NPU_");
        content = std::regex_replace(content, reg, "VPU_");
        // From 4.0.0, driver compiler start to use VPU_ prefix, the old version uses VPUX_ prefix
        if (compilerVersion.major < 4) {
            // Replace VPU_ with VPUX_ for old driver compiler
            std::regex reg("VPU_");
            content = std::regex_replace(content, reg, "VPUX_");
        }
    }
    return "--config " + content;
}

// Parse the result string of query from foramt <name_0><name_1><name_2> to unordered_set of string
static std::unordered_set<std::string> parseQueryResult(std::vector<char>& data) {
    std::string dataString(data.begin(), data.end());
    std::unordered_set<std::string> result;
    size_t i = 0, start = 0;
    while (i < dataString.length()) {
        if (dataString[i] == '<') {
            start = ++i;
        } else if (dataString[i] == '>') {
            std::string temp(dataString.begin() + start, dataString.begin() + i);
            result.insert(temp);
            i++;
        } else {
            i++;
        }
    }
    return result;
}

// For ext version < 1.3, query is unsupported, return empty result and add debug log here
template <typename TableExtension>
template <typename T, std::enable_if_t<NotSupportQuery(T), bool>>
std::unordered_set<std::string> LevelZeroCompilerInDriver<TableExtension>::queryImpl(const std::vector<char>& xml,
                                                                                     const std::vector<char>& weights,
                                                                                     const vpux::Config& config) {
    UNUSED(xml);
    UNUSED(weights);
    UNUSED(config);
    _logger.debug("Driver version is less than 1.3, queryNetwork is unsupported.");
    return std::unordered_set<std::string>();
}

// For ext version >= 1.3, query is supported, calling querynetwork api in _graphDdiTableExt
template <typename TableExtension>
template <typename T, std::enable_if_t<!NotSupportQuery(T), bool>>
std::unordered_set<std::string> LevelZeroCompilerInDriver<TableExtension>::queryImpl(const std::vector<char>& xml,
                                                                                     const std::vector<char>& weights,
                                                                                     const vpux::Config& config) {
    _logger.debug("Calling queryNetwork of 1.3 version.");

    std::string buildFlags;
    ze_device_graph_properties_t deviceGraphProperties{};
    auto result = _graphDdiTableExt->pfnDeviceGetGraphProperties(_deviceHandle, &deviceGraphProperties);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerInDriver: Failed to get graph properties from compiler";
    }
    ze_graph_compiler_version_info_t& compilerVersion = deviceGraphProperties.compilerVersion;
    buildFlags += serializeConfig(config, compilerVersion);
    _logger.debug("Build flags : {0}", buildFlags);

    auto serializedIR = serializeIR(xml, weights, compilerVersion);

    ze_graph_desc_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                            nullptr,
                            ZE_GRAPH_FORMAT_NGRAPH_LITE,
                            serializedIR.size(),
                            serializedIR.data(),
                            buildFlags.c_str()};
    ze_graph_query_network_handle_t hGraphQueryNetwork;

    // Create querynetwork handle
    result = _graphDdiTableExt->pfnQueryNetworkCreate(_context, _deviceHandle, &desc, &hGraphQueryNetwork);

    if (ZE_RESULT_SUCCESS != result) {
        _graphDdiTableExt->pfnQueryNetworkDestroy(hGraphQueryNetwork);
        IE_THROW() << "Failed to Create query network. Error code: " << std::hex << result;
    }

    // Get the size of query result
    size_t size = 0;
    result = _graphDdiTableExt->pfnQueryNetworkGetSupportedLayers(hGraphQueryNetwork, &size, nullptr);
    if (ZE_RESULT_SUCCESS != result) {
        _graphDdiTableExt->pfnQueryNetworkDestroy(hGraphQueryNetwork);
        IE_THROW() << "Failed to get size of querynetwork result. Error code: " << std::hex << result;
    }

    // Get the result data of query
    std::vector<char> supportedLayers(size);
    result = _graphDdiTableExt->pfnQueryNetworkGetSupportedLayers(hGraphQueryNetwork, &size, supportedLayers.data());
    if (ZE_RESULT_SUCCESS != result) {
        _graphDdiTableExt->pfnQueryNetworkDestroy(hGraphQueryNetwork);
        IE_THROW() << "Failed to get data of querynetwork result. Error code: " << std::hex << result;
    }

    result = _graphDdiTableExt->pfnQueryNetworkDestroy(hGraphQueryNetwork);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to destroy graph query network handle. Error code: " << std::hex << result;
    }

    return parseQueryResult(supportedLayers);
}

template <typename TableExtension>
std::unordered_set<std::string> LevelZeroCompilerInDriver<TableExtension>::getQueryResult(
        const std::vector<char>& xml, const std::vector<char>& weights, const vpux::Config& config) {
    _logger.setLevel(config.get<LOG_LEVEL>());
    _logger.debug("LevelZeroCompilerInDriver::getQueryResult");
    auto queryResult = queryImpl(xml, weights, config);
    _logger.debug("LevelZeroCompilerInDriver::getQueryResult end");
    return queryResult;
}

// For ext version <1.5, calling pfnCreate api in _graphDdiTableExt
template <typename TableExtension>
template <typename T, std::enable_if_t<NotSupportGraph2(T), bool>>
ze_result_t LevelZeroCompilerInDriver<TableExtension>::createGraph(const ze_graph_format_t& format,
                                                                   const std::vector<uint8_t>& serializedIR,
                                                                   const std::string& buildFlags, const uint32_t& flags,
                                                                   ze_graph_handle_t* graph) {
    UNUSED(flags);
    ze_graph_desc_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                            nullptr,
                            format,
                            serializedIR.size(),
                            serializedIR.data(),
                            buildFlags.c_str()};

    // Create querynetwork handle
    return _graphDdiTableExt->pfnCreate(_context, _deviceHandle, &desc, graph);
}

// For ext version >= 1.5, calling pfnCreate2 api in _graphDdiTableExt
template <typename TableExtension>
template <typename T, std::enable_if_t<!NotSupportGraph2(T), bool>>
ze_result_t LevelZeroCompilerInDriver<TableExtension>::createGraph(const ze_graph_format_t& format,
                                                                   const std::vector<uint8_t>& serializedIR,
                                                                   const std::string& buildFlags, const uint32_t& flags,
                                                                   ze_graph_handle_t* graph) {
    ze_graph_desc_2_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                              nullptr,
                              format,
                              serializedIR.size(),
                              serializedIR.data(),
                              buildFlags.c_str(),
                              flags};

    // Create querynetwork handle
    return _graphDdiTableExt->pfnCreate2(_context, _deviceHandle, &desc, graph);
}

template <typename TableExtension>
INetworkDescription::Ptr LevelZeroCompilerInDriver<TableExtension>::compileIR(
        const std::string& graphName, const std::vector<char>& xml, const std::vector<char>& weights,
        const ie::InputsDataMap& inputMetadata, const ie::OutputsDataMap& outputMetadata, const vpux::Config& config) {
    _logger.setLevel(config.get<LOG_LEVEL>());
    _logger.debug("LevelZeroCompilerInDriver::compileIR");

    ze_device_graph_properties_t deviceGraphProperties{};
    auto result = _graphDdiTableExt->pfnDeviceGetGraphProperties(_deviceHandle, &deviceGraphProperties);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerInDriver: Failed to get graph properties from compiler";
    }
    ze_graph_compiler_version_info_t& compilerVersion = deviceGraphProperties.compilerVersion;

    auto serializedIR = serializeIR(xml, weights, compilerVersion);

    ze_graph_format_t format = ZE_GRAPH_FORMAT_NGRAPH_LITE;

    std::string buildFlags;

    buildFlags += serializeIOInfo(inputMetadata, outputMetadata);
    buildFlags += " ";
    buildFlags += serializeConfig(config, compilerVersion);

    _logger.debug("Build flags : {0}", buildFlags);
    // TODO #-30202 Store graph_handle inside NetworkDesc instead of blob. But this will require changes in zeroAPI

    // Graph handle should be used only in scope of compile / parse functions.
    ze_graph_handle_t graphHandle;

    // If OV cache is enabled, disable driver caching
    uint32_t flags = ZE_GRAPH_FLAG_NONE;
    const auto set_cache_dir = config.get<CACHE_DIR>();
    if (!set_cache_dir.empty()) {
        flags = flags | ZE_GRAPH_FLAG_DISABLE_CACHING;
    }

    _logger.info("Using extension version: {0}", typeid(TableExtension).name());
    result = createGraph(format, serializedIR, buildFlags, flags, &graphHandle);

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
void LevelZeroCompilerInDriver<TableExtension>::getDeviceIO(NetworkIOVector& devInputs, NetworkIOVector& devOutputs,
                                                            const T& arg) {
    ie::Precision dev_precision = toIEPrecision(arg.devicePrecision);
    ie::Layout devLayout = toIELayout(arg.deviceLayout);
    ie::SizeVector devDims(arg.dims, arg.dims + getDimCount(devLayout));
    ie::TensorDesc devDataDesc(dev_precision, devDims, devLayout);

    if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
        _logger.info("Found input \"{0}\"", arg.name);
        devInputs.emplace_back(arg.name, std::make_shared<ie::Data>(arg.name, devDataDesc));
    }

    if (ZE_GRAPH_ARGUMENT_TYPE_OUTPUT == arg.type) {
        _logger.info("Found output \"{0}\"", arg.name);
        devOutputs.emplace_back(arg.name, std::make_shared<ie::Data>(arg.name, devDataDesc));
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

void getOVNodes(std::vector<OVRawNode>& ovInfo, ze_graph_argument_properties_3_t& arg) {
    ov::Shape nodeShape;
    std::unordered_set<std::string> nodeTensorsNames;
    for (uint32_t id = 0; id < arg.associated_tensor_names_count; id++) {
        nodeTensorsNames.insert(arg.associated_tensor_names[id]);
    }
    for (uint32_t id = 0; id < arg.dims_count; id++) {
        nodeShape.push_back(arg.dims[id]);
    }
    ov::element::Type_t nodeType = toOVElementType(arg.devicePrecision);
    ovInfo.push_back({arg.debug_friendly_name, nodeType, nodeShape, std::move(nodeTensorsNames), arg.name});
}

template <>
void LevelZeroCompilerInDriver<ze_graph_dditable_ext_t>::getMetaData(ze_graph_dditable_ext_t* graphDdiTableExt,
                                                                     ze_graph_handle_t graphHandle, uint32_t index,
                                                                     NetworkIOVector& devInputs,
                                                                     NetworkIOVector& devOutputs,
                                                                     std::vector<OVRawNode>& ovResults,
                                                                     std::vector<OVRawNode>& ovParameters) {
    ze_graph_argument_properties_t arg;
    auto result = graphDdiTableExt->pfnGetArgumentProperties(graphHandle, index, &arg);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerInDriver: LevelZeroCompilerInDriver:Failed to call pfnGetArgumentProperties. "
                      "Error code: "
                   << std::hex << result;
    }

    getDeviceIO(devInputs, devOutputs, arg);
    _logger.warning("ovResults is unsupported, thus stays unchanged.");
    _logger.warning("ovParameters is unsupported, thus stays unchanged.");
    UNUSED(ovResults);
    UNUSED(ovParameters);
}

template <>
void LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_1_t>::getMetaData(ze_graph_dditable_ext_1_1_t* graphDdiTableExt,
                                                                         ze_graph_handle_t graphHandle, uint32_t index,
                                                                         NetworkIOVector& devInputs,
                                                                         NetworkIOVector& devOutputs,
                                                                         std::vector<OVRawNode>& ovResults,
                                                                         std::vector<OVRawNode>& ovParameters) {
    ze_graph_argument_properties_2_t arg;
    auto result = graphDdiTableExt->pfnGetArgumentProperties2(graphHandle, index, &arg);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "LevelZeroCompilerInDriver: Failed to call pfnGetArgumentProperties2. Error code: " << std::hex
                   << result;
    }

    getDeviceIO(devInputs, devOutputs, arg);

    // The I/O data corresponding to the states of the model is not found within the OpenVINO 2.0 attributes contained
    // by the compiled model, thus we should not query them
    if (!isStateInputName(arg.name) && !isStateOutputName(arg.name)) {
        ze_graph_argument_metadata_t metaData;
        result = graphDdiTableExt->pfnGraphGetArgumentMetadata(graphHandle, index, &metaData);
        if (ZE_RESULT_SUCCESS != result) {
            IE_THROW() << "LevelZeroCompilerInDriver: Failed to call pfnGraphGetArgumentMetadata. Error code: "
                       << std::hex << result;
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
void LevelZeroCompilerInDriver<TableExtension>::getMetaData(TableExtension* graphDdiTableExt,
                                                            ze_graph_handle_t graphHandle, uint32_t index,
                                                            NetworkIOVector& devInputs, NetworkIOVector& devOutputs,
                                                            std::vector<OVRawNode>& ovResults,
                                                            std::vector<OVRawNode>& ovParameters) {
    ze_graph_argument_properties_3_t arg;
    auto result = graphDdiTableExt->pfnGetArgumentProperties3(graphHandle, index, &arg);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "pfnGetArgumentProperties3, Failed to get information about inputs/outputs. Error code: "
                   << std::hex << result;
    }
    getDeviceIO(devInputs, devOutputs, arg);
    if (!isStateInputName(arg.name) && !isStateOutputName(arg.name)) {
        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
            getOVNodes(ovParameters, arg);
        }

        if (ZE_GRAPH_ARGUMENT_TYPE_OUTPUT == arg.type) {
            getOVNodes(ovResults, arg);
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

    NetworkIOVector devInputs;
    NetworkIOVector devOutputs;

    std::vector<OVRawNode> ovResults;
    std::vector<OVRawNode> ovParameters;
    for (uint32_t index = 0; index < graphProperties.numGraphArgs; ++index) {
        getMetaData(_graphDdiTableExt, graphHandle, index, devInputs, devOutputs, ovResults, ovParameters);
    }
    // TODO: support this information in CiD [track: E#33479]
    int numStreams = 1;
    return NetworkMeta{std::move(devInputs), std::move(devOutputs), std::move(ovResults), std::move(ovParameters),
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

template class LevelZeroCompilerInDriver<ze_graph_dditable_ext_t>;
template class LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_1_t>;
template class LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_2_t>;
template class LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_3_t>;
template class LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_4_t>;
template class LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_5_t>;

}  // namespace driverCompilerAdapter
}  // namespace vpux
