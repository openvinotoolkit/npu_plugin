//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#include <regex>
#include <string_view>

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"
#include "vpux/utils/IE/itt.hpp"
#include "vpux/utils/IE/prefix.hpp"
#include "zero_compiler_in_driver.h"

#define UNUSED(x) (void)(x)

namespace {

constexpr std::string_view INPUTS_PRECISIONS_KEY = "--inputs_precisions";
constexpr std::string_view INPUTS_LAYOUTS_KEY = "--inputs_layouts";
constexpr std::string_view OUTPUTS_PRECISIONS_KEY = "--outputs_precisions";
constexpr std::string_view OUTPUTS_LAYOUTS_KEY = "--outputs_layouts";

// <option key>="<option value>"
constexpr std::string_view KEY_VALUE_SEPARATOR = "=";
constexpr std::string_view VALUE_DELIMITER = "\"";  // marks beginning and end of value

// Format inside "<option value>"
// <name1>:<value (precision / layout)> [<name2>:<value>]
constexpr std::string_view NAME_VALUE_SEPARATOR = ":";
constexpr std::string_view VALUES_SEPARATOR = " ";

// Constants indicating the order indices needed to be applied as to perform conversions between legacy layout values
const std::vector<size_t> NC_TO_CN_LAYOUT_DIMENSIONS_ORDER = {1, 0};
const std::vector<size_t> NCHW_TO_NHWC_LAYOUT_DIMENSIONS_ORDER = {0, 2, 3, 1};
const std::vector<size_t> NCDHW_TO_NDHWC_LAYOUT_DIMENSIONS_ORDER = {0, 2, 3, 4, 1};

/**
 * @brief A standard copy function concerning memory segments. Additional checks on the given arguments are performed
 * before copying.
 * @details This is meant as a replacement for the legacy "ie_memcpy" function coming from the OpenVINO API.
 */
void checkedMemcpy(void* destination, size_t destinationSize, void const* source, size_t numberOfBytes) {
    if (numberOfBytes == 0) {
        return;
    }

    OPENVINO_ASSERT(destination != nullptr, "Memcpy: received a null destination address");
    OPENVINO_ASSERT(source != nullptr, "Memcpy: received a null source address");
    OPENVINO_ASSERT(numberOfBytes <= destinationSize,
                    "Memcpy: the source buffer does not fit inside the destination one");
    OPENVINO_ASSERT(numberOfBytes <= (destination > source ? ((uintptr_t)destination - (uintptr_t)source)
                                                           : ((uintptr_t)source - (uintptr_t)destination)),
                    "Memcpy: the offset between the two buffers does not allow a safe execution of the operation");

    memcpy(destination, source, numberOfBytes);
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

/**
 * @brief For driver backward compatibility reasons, the given value shall be converted to a string corresponding to the
 * adequate legacy precision.
 */
std::string ovPrecisionToLegacyPrecisionString(const ov::element::Type& precision) {
    switch (precision) {
    case ov::element::Type_t::undefined:
        return "UNSPECIFIED";
    case ov::element::Type_t::f16:
        return "FP16";
    case ov::element::Type_t::f32:
        return "FP32";
    case ov::element::Type_t::f64:
        return "FP64";
    case ov::element::Type_t::bf16:
        return "BF16";
    case ov::element::Type_t::i4:
        return "I4";
    case ov::element::Type_t::i8:
        return "I8";
    case ov::element::Type_t::i16:
        return "I16";
    case ov::element::Type_t::i32:
        return "I32";
    case ov::element::Type_t::i64:
        return "I64";
    case ov::element::Type_t::u4:
        return "U4";
    case ov::element::Type_t::u8:
        return "U8";
    case ov::element::Type_t::u16:
        return "U16";
    case ov::element::Type_t::u32:
        return "U32";
    case ov::element::Type_t::u64:
        return "U64";
    case ov::element::Type_t::u1:
        return "BIN";
    case ov::element::Type_t::boolean:
        return "BOOL";
    case ov::element::Type_t::dynamic:
        return "DYNAMIC";
    default:
        OPENVINO_THROW("Incorrect precision: ", precision);
    }
}

/**
 * @brief Gives the string representation of the default legacy layout value corresponding to the given rank.
 * @details This is done in order to assure the backward compatibility with the driver. Giving a layout different from
 * the default one may lead either to error or to accuracy failures since unwanted transposition layers may be
 * introduced.
 */
std::string rankToLegacyLayoutString(const size_t rank) {
    switch (rank) {
    case 0:
        return "**SCALAR**";
    case 1:
        return "C";
    case 2:
        return "NC";
    case 3:
        return "CHW";
    case 4:
        return "NCHW";
    case 5:
        return "NCDHW";
    default:
        return "BLOCKED";
    }
}

size_t zeLayoutToRank(const ze_graph_argument_layout_t layout) {
    switch (layout) {
    case ZE_GRAPH_ARGUMENT_LAYOUT_C:
        return 1;
    case ZE_GRAPH_ARGUMENT_LAYOUT_CN:
        return 2;
    case ZE_GRAPH_ARGUMENT_LAYOUT_HW:
        return 2;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NC:
        return 2;
    case ZE_GRAPH_ARGUMENT_LAYOUT_CHW:
        return 3;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCHW:
        return 4;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NHWC:
        return 4;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW:
        return 5;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC:
        return 5;
    default:
        // TODO #-30200 Extend to support all cases
        return 0;
    }
}

/**
 * @brief Transposes the original shape value according to given layout.
 */
std::vector<size_t> reshapeByLayout(const std::vector<size_t>& originalDimensions,
                                    const ze_graph_argument_layout_t layout) {
    std::vector<size_t> order;
    std::vector<size_t> reshapedDimensions;

    switch (layout) {
    case ZE_GRAPH_ARGUMENT_LAYOUT_CN:
        order = NC_TO_CN_LAYOUT_DIMENSIONS_ORDER;
        break;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NHWC:
        order = NCHW_TO_NHWC_LAYOUT_DIMENSIONS_ORDER;
        break;
    case ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC:
        order = NCDHW_TO_NDHWC_LAYOUT_DIMENSIONS_ORDER;
        break;
    default:
        // TODO #-30200 Extend to support all cases
        return originalDimensions;
    }

    for (const size_t& orderElement : order) {
        reshapedDimensions.push_back(originalDimensions[orderElement]);
    }

    return reshapedDimensions;
}

}  // namespace

namespace vpux {
namespace driverCompilerAdapter {

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

    OPENVINO_ASSERT(numberOfInputData < maxNumberOfElements);
    if (xmlSize >= maxSizeOfXML) {
        OPENVINO_THROW("LevelZeroCompilerInDriver: Xml file is too big to process.");
    }
    if (weightsSize >= maxSizeOfWeights) {
        OPENVINO_THROW("LevelZeroCompilerInDriver: Bin file is too big to process.");
    }

    const uint64_t sizeOfSerializedIR = sizeof(compilerVersion) + sizeof(numberOfInputData) + sizeof(xmlSize) +
                                        xml.size() + sizeof(weightsSize) + weights.size();

    std::vector<uint8_t> serializedIR;
    serializedIR.resize(sizeOfSerializedIR);

    uint64_t offset = 0;
    checkedMemcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &compilerVersion, sizeof(compilerVersion));
    offset += sizeof(compilerVersion);

    checkedMemcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &numberOfInputData,
                  sizeof(numberOfInputData));
    offset += sizeof(numberOfInputData);
    checkedMemcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &xmlSize, sizeof(xmlSize));
    offset += sizeof(xmlSize);
    checkedMemcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, xml.data(), xmlSize);
    offset += xmlSize;
    checkedMemcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, &weightsSize, sizeof(weightsSize));
    offset += sizeof(weightsSize);
    checkedMemcpy(serializedIR.data() + offset, sizeOfSerializedIR - offset, weights.data(), weightsSize);
    offset += weightsSize;

    OPENVINO_ASSERT(offset == sizeOfSerializedIR);

    return serializedIR;
}

template <typename TableExtension>
std::string LevelZeroCompilerInDriver<TableExtension>::serializeIOInfo(const std::shared_ptr<ov::Model> model) {
    const ov::ParameterVector& parameters = model->get_parameters();
    const ov::ResultVector& results = model->get_results();

    const std::string& firstInputName = parameters.at(0)->get_friendly_name();
    const std::string& firstOutputName = results.at(0)->get_input_node_ptr(0)->get_friendly_name();

    std::stringstream inputsPrecisionSS;
    std::stringstream inputsLayoutSS;
    std::stringstream outputsPrecisionSS;
    std::stringstream outputsLayoutSS;

    inputsPrecisionSS << INPUTS_PRECISIONS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;
    inputsLayoutSS << INPUTS_LAYOUTS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;

    for (const std::shared_ptr<ov::op::v0::Parameter>& parameter : parameters) {
        const std::string& name = parameter->get_friendly_name();
        const ov::element::Type& precision = parameter->get_element_type();
        const size_t rank = parameter->get_shape().size();

        if (name != firstInputName) {
            inputsPrecisionSS << VALUES_SEPARATOR;
            inputsLayoutSS << VALUES_SEPARATOR;
        }

        inputsPrecisionSS << name << NAME_VALUE_SEPARATOR << ovPrecisionToLegacyPrecisionString(precision);
        // Ticket: E-88902
        inputsLayoutSS << name << NAME_VALUE_SEPARATOR << rankToLegacyLayoutString(rank);
    }

    inputsPrecisionSS << VALUE_DELIMITER;
    inputsLayoutSS << VALUE_DELIMITER;

    outputsPrecisionSS << OUTPUTS_PRECISIONS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;
    outputsLayoutSS << OUTPUTS_LAYOUTS_KEY << KEY_VALUE_SEPARATOR << VALUE_DELIMITER;

    for (const std::shared_ptr<ov::op::v0::Result>& result : results) {
        const std::string& name = result->get_input_node_ptr(0)->get_friendly_name();
        const ov::element::Type_t precision = result->get_element_type();
        const size_t rank = result->get_shape().size();

        if (name != firstOutputName) {
            outputsPrecisionSS << VALUES_SEPARATOR;
            outputsLayoutSS << VALUES_SEPARATOR;
        }

        outputsPrecisionSS << name << NAME_VALUE_SEPARATOR << ovPrecisionToLegacyPrecisionString(precision);
        outputsLayoutSS << name << NAME_VALUE_SEPARATOR << rankToLegacyLayoutString(rank);
    }

    outputsPrecisionSS << VALUE_DELIMITER;
    outputsLayoutSS << VALUE_DELIMITER;

    // One line without spaces to avoid parsing as config option inside CID
    return inputsPrecisionSS.str() + VALUES_SEPARATOR.data() + inputsLayoutSS.str() + VALUES_SEPARATOR.data() +
           outputsPrecisionSS.str() + VALUES_SEPARATOR.data() + outputsLayoutSS.str();
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

    // As a consequence of complying to the conventions established in the 2.0 OV API, the set of values corresponding
    // to the "model priority" key has been modified
    if ((compilerVersion.major < 5) || (compilerVersion.major == 5 && compilerVersion.minor < 2)) {
        const auto& getTargetRegex = [](const ov::hint::Priority& priorityValue) -> std::regex {
            std::ostringstream result;
            result << ov::hint::model_priority.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER << priorityValue
                   << VALUE_DELIMITER;
            return std::regex(result.str());
        };
        const auto& getStringReplacement = [](const ov::intel_vpux::LegacyPriority& priorityValue) -> std::string {
            std::ostringstream result;
            result << ov::intel_vpux::legacy_model_priority.name() << KEY_VALUE_SEPARATOR << VALUE_DELIMITER
                   << priorityValue << VALUE_DELIMITER;
            return result.str();
        };

        // E.g. (valid as of writing this): MODEL_PRIORITY="MEDIUM" -> MODEL_PRIORITY="MODEL_PRIORITY_MED"
        content = std::regex_replace(content, getTargetRegex(ov::hint::Priority::LOW),
                                     getStringReplacement(ov::intel_vpux::LegacyPriority::LOW));
        content = std::regex_replace(content, getTargetRegex(ov::hint::Priority::MEDIUM),
                                     getStringReplacement(ov::intel_vpux::LegacyPriority::MEDIUM));
        content = std::regex_replace(content, getTargetRegex(ov::hint::Priority::HIGH),
                                     getStringReplacement(ov::intel_vpux::LegacyPriority::HIGH));
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

// For ext version == 1.3 && == 1.4, query is supported, calling querynetwork api in _graphDdiTableExt
template <typename TableExtension>
template <typename T, std::enable_if_t<SupportAPIGraphQueryNetworkV1(T), bool>>
std::unordered_set<std::string> LevelZeroCompilerInDriver<TableExtension>::queryImpl(const std::vector<char>& xml,
                                                                                     const std::vector<char>& weights,
                                                                                     const vpux::Config& config) {
    _logger.debug("Calling queryNetwork of 1.3 version.");

    std::string buildFlags;
    auto serializedIR = getSerializedIR(buildFlags, xml, weights, config);

    ze_graph_desc_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                            nullptr,
                            ZE_GRAPH_FORMAT_NGRAPH_LITE,
                            serializedIR.size(),
                            serializedIR.data(),
                            buildFlags.c_str()};
    ze_graph_query_network_handle_t hGraphQueryNetwork = nullptr;

    // Create querynetwork handle
    auto result = _graphDdiTableExt->pfnQueryNetworkCreate(_context, _deviceHandle, &desc, &hGraphQueryNetwork);

    return getQueryResultFromSupportedLayers(result, hGraphQueryNetwork);
}

// For ext version >= 1.5
template <typename TableExtension>
template <typename T, std::enable_if_t<SupportAPIGraphQueryNetworkV2(T), bool>>
std::unordered_set<std::string> LevelZeroCompilerInDriver<TableExtension>::queryImpl(const std::vector<char>& xml,
                                                                                     const std::vector<char>& weights,
                                                                                     const vpux::Config& config) {
    _logger.debug("Calling queryNetwork of 1.5 version.");

    std::string buildFlags;
    auto serializedIR = getSerializedIR(buildFlags, xml, weights, config);

    ze_graph_desc_2_t desc = {ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES,
                              nullptr,
                              ZE_GRAPH_FORMAT_NGRAPH_LITE,
                              serializedIR.size(),
                              serializedIR.data(),
                              buildFlags.c_str(),
                              ZE_GRAPH_FLAG_NONE};

    ze_graph_query_network_handle_t hGraphQueryNetwork = nullptr;

    // Create querynetwork handle
    auto result = _graphDdiTableExt->pfnQueryNetworkCreate2(_context, _deviceHandle, &desc, &hGraphQueryNetwork);

    return getQueryResultFromSupportedLayers(result, hGraphQueryNetwork);
}

template <typename TableExtension>
template <typename T, std::enable_if_t<!NotSupportQuery(T), bool>>
std::unordered_set<std::string> LevelZeroCompilerInDriver<TableExtension>::getQueryResultFromSupportedLayers(
        ze_result_t result, ze_graph_query_network_handle_t& hGraphQueryNetwork) {
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("Failed to Create query network. Error code: ", std::hex, result);
    }

    // Get the size of query result
    size_t size = 0;
    result = _graphDdiTableExt->pfnQueryNetworkGetSupportedLayers(hGraphQueryNetwork, &size, nullptr);
    if (ZE_RESULT_SUCCESS != result) {
        _graphDdiTableExt->pfnQueryNetworkDestroy(hGraphQueryNetwork);
        OPENVINO_THROW("Failed to get size of querynetwork result. Error code: ", std::hex, result);
    }

    // Get the result data of query
    std::vector<char> supportedLayers(size);
    result = _graphDdiTableExt->pfnQueryNetworkGetSupportedLayers(hGraphQueryNetwork, &size, supportedLayers.data());
    if (ZE_RESULT_SUCCESS != result) {
        _graphDdiTableExt->pfnQueryNetworkDestroy(hGraphQueryNetwork);
        OPENVINO_THROW("Failed to get data of querynetwork result. Error code: ", std::hex, result);
    }

    result = _graphDdiTableExt->pfnQueryNetworkDestroy(hGraphQueryNetwork);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("Failed to destroy graph query network handle. Error code: ", std::hex, result);
    }

    return parseQueryResult(supportedLayers);
}

template <typename TableExtension>
std::vector<uint8_t> LevelZeroCompilerInDriver<TableExtension>::getSerializedIR(std::string& buildFlags,
                                                                                const std::vector<char>& xml,
                                                                                const std::vector<char>& weights,
                                                                                const vpux::Config& config) {
    ze_device_graph_properties_t deviceGraphProperties{};
    auto result = _graphDdiTableExt->pfnDeviceGetGraphProperties(_deviceHandle, &deviceGraphProperties);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("LevelZeroCompilerInDriver: Failed to get graph properties from compiler");
    }
    ze_graph_compiler_version_info_t& compilerVersion = deviceGraphProperties.compilerVersion;
    buildFlags += serializeConfig(config, compilerVersion);
    _logger.debug("Build flags : {0}", buildFlags);

    auto serializedIR = serializeIR(xml, weights, compilerVersion);

    return serializedIR;
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
INetworkDescription::Ptr LevelZeroCompilerInDriver<TableExtension>::compileIR(const std::shared_ptr<ov::Model> model,
                                                                              const std::string& graphName,
                                                                              const std::vector<char>& xml,
                                                                              const std::vector<char>& weights,
                                                                              const vpux::Config& config) {
    _logger.setLevel(config.get<LOG_LEVEL>());
    _logger.debug("LevelZeroCompilerInDriver::compileIR");

    ze_device_graph_properties_t deviceGraphProperties{};
    auto result = _graphDdiTableExt->pfnDeviceGetGraphProperties(_deviceHandle, &deviceGraphProperties);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("LevelZeroCompilerInDriver: Failed to get graph properties from compiler");
    }
    ze_graph_compiler_version_info_t& compilerVersion = deviceGraphProperties.compilerVersion;

    auto serializedIR = serializeIR(xml, weights, compilerVersion);

    ze_graph_format_t format = ZE_GRAPH_FORMAT_NGRAPH_LITE;

    std::string buildFlags;

    buildFlags += serializeIOInfo(model);
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
        OPENVINO_THROW("LevelZeroCompilerInDriver: Failed to destroy graph handle. Error code: ", std::hex, result);
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
            OPENVINO_THROW("LevelZeroCompilerInDriver: Failed to import blob. Error code: ", std::hex, result);
        }
    } else {
        OPENVINO_THROW("Empty blob");
    }

    OV_ITT_TASK_NEXT(PARSE_BLOB, "getNetworkMeta");
    const auto networkMeta = getNetworkMeta(graphHandle);
    OV_ITT_TASK_NEXT(PARSE_BLOB, "NetworkDescription");

    auto result = _graphDdiTableExt->pfnDestroy(graphHandle);

    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("LevelZeroCompilerInDriver: Failed to destroy graph handle. Error code: ", std::hex, result);
    }

    return std::make_shared<driverCompilerAdapter::NetworkDescription>(blob, graphName, networkMeta);
}

template <typename TableExtension>
uint32_t LevelZeroCompilerInDriver<TableExtension>::getSupportedOpset() {
    _logger.debug("LevelZeroCompilerInDriver::getSupportedOpset");
    ze_device_graph_properties_t graphProperties;

    auto result = _graphDdiTableExt->pfnDeviceGetGraphProperties(_deviceHandle, &graphProperties);

    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("LevelZeroCompilerInDriver: Failed to get opset version from compiler");
    }
    const auto maxOpsetVersion = graphProperties.maxOVOpsetVersionSupported;
    _logger.info("Max supported version of opset in CiD: {0}", maxOpsetVersion);
    return maxOpsetVersion;
}

template <typename TableExtension>
template <typename T>
void LevelZeroCompilerInDriver<TableExtension>::getLayoutOrStateDescriptor(IONodeDescriptorMap& parameters,
                                                                           IONodeDescriptorMap& results,
                                                                           IONodeDescriptorMap& states,
                                                                           std::vector<std::string>& stateNames,
                                                                           const T& arg) {
    std::string legacyName = arg.name;

    // The layout may differ from the default one only when using significantly older drivers. In order to accommodate
    // this case, an extra attribute needs to be stored which holds the transposed shape.
    const std::vector<size_t> originalDimensions(arg.dims, arg.dims + zeLayoutToRank(arg.deviceLayout));
    const std::vector<size_t> reshapedDimensions = reshapeByLayout(originalDimensions, arg.deviceLayout);
    const ov::Shape shape = ov::Shape(reshapedDimensions);

    if (!isStateInputName(legacyName) && !isStateOutputName(legacyName)) {
        if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            _logger.info("Found input \"{0}\"", legacyName);

            parameters[legacyName].transposedShape = shape;
        }
        if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_OUTPUT) {
            _logger.info("Found output \"{0}\"", legacyName);

            results[legacyName].transposedShape = shape;
        }
    } else if (isStateInputName(legacyName)) {
        // The inputs and outputs of the state nodes share the same metadata, thus we'll consider only the the inputs
        // here
        legacyName = legacyName.substr(READVALUE_PREFIX.length());
        _logger.info("Found state variable \"{0}\"", legacyName);

        const ov::element::Type_t precision = toOVElementType(arg.devicePrecision);

        stateNames.push_back(legacyName);
        states[legacyName] = {legacyName, "", {}, precision, shape, shape};
    }
}

template <typename TableExtension>
template <typename T, std::enable_if_t<std::is_same<T, ze_graph_dditable_ext_t>::value, bool>>
void LevelZeroCompilerInDriver<TableExtension>::getNodeOrStateDescriptorLegacy(
        IONodeDescriptorMap& parameters, IONodeDescriptorMap& results, IONodeDescriptorMap& states,
        std::vector<std::string>& inputNames, std::vector<std::string>& outputNames,
        std::vector<std::string>& stateNames, const ze_graph_argument_properties_t& arg) {
    std::string legacyName = arg.name;
    const ov::element::Type_t precision = toOVElementType(arg.devicePrecision);

    // The layout shall differ from the default one only when using significantly older drivers. In order to accommodate
    // this case, an extra attribute needs to be stored which holds the transposed shape.
    const std::vector<size_t> originalDimensions(arg.dims, arg.dims + zeLayoutToRank(arg.deviceLayout));
    const std::vector<size_t> reshapedDimensions = reshapeByLayout(originalDimensions, arg.deviceLayout);
    const ov::Shape originalShape = ov::Shape(originalDimensions);
    const ov::Shape transposedShape = ov::Shape(reshapedDimensions);

    if (!isStateInputName(legacyName) && !isStateOutputName(legacyName)) {
        if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT) {
            _logger.info("Found input \"{0}\"", legacyName);

            inputNames.push_back(legacyName);
            parameters[legacyName] = {legacyName, legacyName, {legacyName}, precision, originalShape, transposedShape};
        }
        if (arg.type == ZE_GRAPH_ARGUMENT_TYPE_OUTPUT) {
            _logger.info("Found output \"{0}\"", legacyName);

            outputNames.push_back(legacyName);
            results[legacyName] = {legacyName, legacyName, {legacyName}, precision, originalShape, transposedShape};
        }
    } else if (isStateInputName(legacyName)) {
        // The inputs and outputs of the state nodes share the same metadata, thus we'll consider only the the inputs
        // here
        legacyName = legacyName.substr(READVALUE_PREFIX.length());
        _logger.info("Found state variable \"{0}\"", legacyName);

        stateNames.push_back(legacyName);
        states[legacyName] = {legacyName, "", {}, precision, originalShape, originalShape};
    }
}

/**
 * @brief Extracts the parameter/result (i.e. input/output) descriptors from Level Zero specific structures into
 * OpenVINO specific ones.
 * @param nodeDescriptors The map in which the result shall be stored.
 * @param names The I/O identifiers shall be stored here in the order found within the compiled model.
 * @param metadata The Level Zero structure fomr which the descriptors will be extracted.
 */
void getNodeDescriptor(IONodeDescriptorMap& nodeDescriptors, std::vector<std::string>& names,
                       ze_graph_argument_metadata_t& metadata) {
    const ov::element::Type_t precision = toOVElementType(metadata.data_type);
    ov::Shape shape;
    std::unordered_set<std::string> outputTensorNames;

    for (uint32_t id = 0; id < metadata.tensor_names_count; id++) {
        outputTensorNames.insert(metadata.tensor_names[id]);
    }
    for (uint32_t id = 0; id < metadata.shape_size; id++) {
        shape.push_back(metadata.shape[id]);
    }
    const std::string& legacyName = metadata.input_name;

    names.push_back(legacyName);
    nodeDescriptors[legacyName] = {legacyName, metadata.friendly_name, std::move(outputTensorNames), precision, shape,
                                   shape};
}

void getNodeDescriptor(IONodeDescriptorMap& nodeDescriptors, std::vector<std::string>& names,
                       ze_graph_argument_properties_3_t& arg) {
    ov::element::Type_t precision = toOVElementType(arg.devicePrecision);
    ov::Shape shape;
    std::unordered_set<std::string> outputTensorNames;

    for (uint32_t id = 0; id < arg.associated_tensor_names_count; id++) {
        outputTensorNames.insert(arg.associated_tensor_names[id]);
    }
    for (uint32_t id = 0; id < arg.dims_count; id++) {
        shape.push_back(arg.dims[id]);
    }
    const std::string& legacyName = arg.name;

    names.push_back(legacyName);
    nodeDescriptors[legacyName] = {legacyName, arg.debug_friendly_name, std::move(outputTensorNames), precision, shape,
                                   shape};
}

template <>
void LevelZeroCompilerInDriver<ze_graph_dditable_ext_t>::getMetadata(
        ze_graph_dditable_ext_t* graphDdiTableExt, ze_graph_handle_t graphHandle, uint32_t index,
        std::vector<std::string>& inputNames, std::vector<std::string>& outputNames,
        std::vector<std::string>& stateNames, IONodeDescriptorMap& parameters, IONodeDescriptorMap& results,
        IONodeDescriptorMap& states) {
    ze_graph_argument_properties_t arg;
    auto result = graphDdiTableExt->pfnGetArgumentProperties(graphHandle, index, &arg);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("LevelZeroCompilerInDriver: LevelZeroCompilerInDriver:Failed to call pfnGetArgumentProperties. "
                       "Error code: ",
                       std::hex, result);
    }

    getNodeOrStateDescriptorLegacy(parameters, results, states, inputNames, outputNames, stateNames, arg);
}

template <>
void LevelZeroCompilerInDriver<ze_graph_dditable_ext_1_1_t>::getMetadata(
        ze_graph_dditable_ext_1_1_t* graphDdiTableExt, ze_graph_handle_t graphHandle, uint32_t index,
        std::vector<std::string>& inputNames, std::vector<std::string>& outputNames,
        std::vector<std::string>& stateNames, IONodeDescriptorMap& parameters, IONodeDescriptorMap& results,
        IONodeDescriptorMap& states) {
    ze_graph_argument_properties_2_t arg;
    auto result = graphDdiTableExt->pfnGetArgumentProperties2(graphHandle, index, &arg);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("LevelZeroCompilerInDriver: Failed to call pfnGetArgumentProperties2. Error code: ", std::hex,
                       result);
    }

    // The I/O data corresponding to the states of the model is not found within the OpenVINO 2.0 attributes contained
    // by the compiled model, thus we should not query them
    if (!isStateInputName(arg.name) && !isStateOutputName(arg.name)) {
        ze_graph_argument_metadata_t metadata;
        result = graphDdiTableExt->pfnGraphGetArgumentMetadata(graphHandle, index, &metadata);
        if (ZE_RESULT_SUCCESS != result) {
            OPENVINO_THROW("LevelZeroCompilerInDriver: Failed to call pfnGraphGetArgumentMetadata. Error code: ",
                           std::hex, result);
        }

        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
            getNodeDescriptor(parameters, inputNames, metadata);
        }
        if (ZE_GRAPH_ARGUMENT_TYPE_OUTPUT == arg.type) {
            getNodeDescriptor(results, outputNames, metadata);
        }
    }

    getLayoutOrStateDescriptor(parameters, results, states, stateNames, arg);
}

template <typename TableExtension>
void LevelZeroCompilerInDriver<TableExtension>::getMetadata(TableExtension* graphDdiTableExt,
                                                            ze_graph_handle_t graphHandle, uint32_t index,
                                                            std::vector<std::string>& inputNames,
                                                            std::vector<std::string>& outputNames,
                                                            std::vector<std::string>& stateNames,
                                                            IONodeDescriptorMap& parameters,
                                                            IONodeDescriptorMap& results, IONodeDescriptorMap& states) {
    ze_graph_argument_properties_3_t arg;
    auto result = graphDdiTableExt->pfnGetArgumentProperties3(graphHandle, index, &arg);
    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("pfnGetArgumentProperties3, Failed to get information about inputs/outputs. Error code: ",
                       std::hex, result);
    }

    if (!isStateInputName(arg.name) && !isStateOutputName(arg.name)) {
        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
            getNodeDescriptor(parameters, inputNames, arg);
        }

        if (ZE_GRAPH_ARGUMENT_TYPE_OUTPUT == arg.type) {
            getNodeDescriptor(results, outputNames, arg);
        }
    }

    getLayoutOrStateDescriptor(parameters, results, states, stateNames, arg);
}

template <typename TableExtension>
NetworkMeta LevelZeroCompilerInDriver<TableExtension>::getNetworkMeta(ze_graph_handle_t graphHandle) {
    ze_graph_properties_t graphProperties{};

    auto result = _graphDdiTableExt->pfnGetProperties(graphHandle, &graphProperties);

    if (ZE_RESULT_SUCCESS != result) {
        OPENVINO_THROW("LevelZeroCompilerInDriver: Failed to get information about graph.");
    }

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<std::string> stateNames;

    IONodeDescriptorMap parameters;
    IONodeDescriptorMap results;
    IONodeDescriptorMap states;

    for (uint32_t index = 0; index < graphProperties.numGraphArgs; ++index) {
        getMetadata(_graphDdiTableExt, graphHandle, index, inputNames, outputNames, stateNames, parameters, results,
                    states);
    }
    // TODO: support this information in CiD [track: E#33479]
    int numStreams = 1;
    return NetworkMeta{std::move(inputNames),
                       std::move(outputNames),
                       std::move(stateNames),
                       std::move(parameters),
                       std::move(results),
                       std::move(states),
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
