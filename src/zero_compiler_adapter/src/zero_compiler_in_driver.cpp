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

#include "zero_compiler_in_driver.h"
#include <chrono>
#include "ie_layouts.h"
#include "network_description.h"

namespace vpux {
namespace zeroCompilerAdapter {

namespace IE = InferenceEngine;
//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
namespace {
using SerializedIR = std::vector<uint8_t>;
/**
 * @brief Place xml + weights in sequential memory
 * @details Format of the memory:
 *  1. Number of data element (now only xml + weights = 2)
 *  2. Size of data 1 (xml)
 *  3. Data 1
 *  4. Size of data 2 (weights)
 *  5. Data 2
 */

const uint32_t maxNumberOfElements = 10;
const uint64_t maxSizeOfXML = std::numeric_limits<uint64_t>::max() / 3;
const uint64_t maxSizeOfWeights = maxSizeOfXML * 2;

SerializedIR serializeIR(const std::vector<char>& xml, const std::vector<char>& weights) {
    const uint32_t numberOfInputData = 2;
    const uint64_t xmlSize = static_cast<uint64_t>(xml.size());
    const uint64_t weightsSize = static_cast<uint64_t>(weights.size());

    // TODO Refactor checks for correct size
    IE_ASSERT(numberOfInputData < maxNumberOfElements);
    IE_ASSERT(xmlSize < maxSizeOfXML);
    IE_ASSERT(weightsSize < maxSizeOfWeights);

    const size_t sizeOfSerializedIR =
            sizeof(numberOfInputData) + sizeof(xmlSize) + xml.size() + sizeof(weightsSize) + weights.size();

    std::vector<uint8_t> serializedIR;
    serializedIR.resize(sizeOfSerializedIR);

    uint64_t offset = 0;
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
}  // namespace

// TODO Not all Precision from IE listed in ze_graph_ext
IE::Precision toIEPrecision(const ze_graph_argument_precision_t ze_precision) {
    switch (ze_precision) {
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

IE::Layout toIELayout(const ze_graph_argument_layout_t ze_layout) {
    switch (ze_layout) {
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
        // TODO Extend to support all cases
        return 0;
    }

    return 0;
}

//------------------------------------------------------------------------------
// TODO Use log level to logger from config or global logger value
// TODO Fix log messages and error handling. Throw instead of just return, since no error handling in return case.
// TODO Copy-paste form adapter for most case
LevelZeroCompilerInDriver::LevelZeroCompilerInDriver(): _logger("LevelZeroCompilerInDriver", LogLevel::Debug) {
    _logger.debug("LevelZeroCompilerInDriver::LevelZeroCompilerInDriver");
    auto result = zeInit(0);
    if (ZE_RESULT_SUCCESS != result) {
        std::cerr << "ZeroDevicesSingleton zeInit failed 0x" << std::hex << uint64_t(result) << std::dec << std::endl;
        return;
    }

    uint32_t drivers = 0;
    result = zeDriverGet(&drivers, nullptr);
    if (ZE_RESULT_SUCCESS != result) {
        std::cerr << "ZeroDevicesSingleton zeDriverGet count failed 0x" << std::hex << uint64_t(result) << std::dec
                  << std::endl;
        return;
    }
    std::vector<ze_driver_handle_t> all_drivers(drivers);
    result = zeDriverGet(&drivers, all_drivers.data());
    if (ZE_RESULT_SUCCESS != result) {
        std::cerr << "ZeroDevicesSingleton zeDriverGet get failed 0x" << std::hex << uint64_t(result) << std::dec
                  << std::endl;
        return;
    }
    // Get our target driver
    for (uint32_t i = 0; i < drivers; ++i) {
        // arbitrary test at this point
        if (i == drivers - 1) {
            _driver_handle = all_drivers[i];
        }
    }

    // Load our graph extension
    result = zeDriverGetExtensionFunctionAddress(_driver_handle, "ZE_extension_graph",
                                                 reinterpret_cast<void**>(&_graph_ddi_table_ext));
    if (ZE_RESULT_SUCCESS != result) {
        std::cerr << "ZeroDevicesSingleton zeDriverGetExtensionFunctionAddress failed 0x" << std::hex
                  << uint64_t(result) << std::dec << std::endl;
        return;
    }

    uint32_t device_count = 1;
    // Get our target device
    result = zeDeviceGet(_driver_handle, &device_count, &_device_handle);
    if (ZE_RESULT_SUCCESS != result) {
        std::cerr << "ZeroDevicesSingleton zeDeviceGet failed 0x" << std::hex << uint64_t(result) << std::dec
                  << std::endl;
        return;
    }

    // Context part removed

    _logger.debug("LevelZeroCompilerInDriver::LevelZeroCompilerInDriver done");
}

// TODO Do we need to do anything here?
LevelZeroCompilerInDriver::~LevelZeroCompilerInDriver() {
    if (_context) {
        auto result = zeContextDestroy(_context);
        if (ZE_RESULT_SUCCESS != result) {
            _logger.warning("zeContextDestroy failed {0:X+}", uint64_t(result));
        }
    }
    _logger.debug("LevelZeroCompilerInDriver obj destroyed");
}

INetworkDescription::Ptr LevelZeroCompilerInDriver::compileIR(const std::string& graphName,
                                                              const std::vector<char>& xml,
                                                              const std::vector<char>& weights) {
    _logger.debug("LevelZeroCompilerInDriver::compileIR");
    auto serializedIR = serializeIR(xml, weights);

    _logger.debug("serializedIR.size() {}", serializedIR.size());
    _logger.debug("serializedIR.data() {}", serializedIR.data());
    ze_graph_format_t format = ZE_GRAPH_FORMAT_NGRAPH_LITE;
    ze_graph_desc_t desc = {format, serializedIR.size(), serializedIR.data()};

    // TODO Store graph_handle inside NetworkDesc instead of blob. But this will require changes in zeroAPI
    // Graph handle should be used only in scope of compile / parse functions.
    ze_graph_handle_t graph_handle;
    auto result = _graph_ddi_table_ext->pfnCreate(_device_handle, &desc, &graph_handle);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to compile graph with zero API";
    }

    // Get blob size first
    size_t blobSize = -1;
    result = _graph_ddi_table_ext->pfnGetNativeBinary(graph_handle, &blobSize, nullptr);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get blob size";
    }

    std::vector<char> blob;
    blob.resize(blobSize);
    // Get blob data
    result = _graph_ddi_table_ext->pfnGetNativeBinary(graph_handle, &blobSize, reinterpret_cast<uint8_t*>(blob.data()));
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get native binary";
    }

    ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
    result = zeContextCreate(_driver_handle, &context_desc, &_context);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to create context";
    }

    const auto networkMeta = getNetworkMeta(graph_handle);
    _logger.debug("LevelZeroCompilerInDriver::compileIR end");
    return std::make_shared<zeroCompilerAdapter::NetworkDescription>(blob, graphName, networkMeta);
}

std::shared_ptr<INetworkDescription> LevelZeroCompilerInDriver::parseBlob(const std::string& graphName,
                                                                          const std::vector<char>& blob) {
    _logger.debug("LevelZeroCompilerInDriver::getNetworkMeta");
    ze_graph_handle_t graph_handle;

    if (!blob.empty()) {
        _logger.debug("Import network case");
        ze_graph_format_t format = ZE_GRAPH_FORMAT_NATIVE;
        ze_graph_desc_t desc = {format, blob.size(), reinterpret_cast<const uint8_t*>(blob.data())};

        auto result = _graph_ddi_table_ext->pfnCreate(_device_handle, &desc, &graph_handle);
        if (ZE_RESULT_SUCCESS != result) {
            IE_THROW() << "Failed to load blob with zero API";
        }
        _logger.debug("compileIRReturnHandle handle: {}", graph_handle);
    } else {
        // TODO Extend handling or log in this case. Also no tests for this?
        THROW_IE_EXCEPTION << "Empty blob";
    }
    const auto networkMeta = getNetworkMeta(graph_handle);
    return std::make_shared<zeroCompilerAdapter::NetworkDescription>(blob, graphName, networkMeta);
}

size_t LevelZeroCompilerInDriver::getSupportedOpset() {
    _logger.debug("LevelZeroCompilerInDriver::getSupportedOpset");
    ze_device_graph_properties_t graph_properties;
    auto result = _graph_ddi_table_ext->pfnDeviceGetGraphProperties(_device_handle, &graph_properties);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get opset version from compiler";
    }
    const auto maxOpsetVersion = graph_properties.maxOVOpsetVersionSupported;
    _logger.info("Max supported version of opset in CiD: {}", maxOpsetVersion);
    return maxOpsetVersion;
}

std::tuple<const NetworkInputs, const NetworkOutputs, const DeviceInputs, const DeviceOutputs>
LevelZeroCompilerInDriver::getNetworkMeta(ze_graph_handle_t graph_handle) {
    ze_graph_properties_t graph_properties{};
    auto result = _graph_ddi_table_ext->pfnGetProperties(graph_handle, &graph_properties);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get pfnGetProperties";
    }

    DataMap net_inputs;
    DataMap dev_inputs;

    DataMap net_outputs;
    DataMap dev_outputs;

    for (uint32_t index = 0; index < graph_properties.numGraphArgs; ++index) {
        ze_graph_argument_properties2_t arg;
        result = _graph_ddi_table_ext->pfnGetArgumentProperties2(graph_handle, index, &arg);
        if (ZE_RESULT_SUCCESS != result) {
            IE_THROW() << "Failed to get pfnGetArgumentProperties2";
        }

        // ze_graph_argument_properties_t arg;
        // result = _graph_ddi_table_ext->pfnGetArgumentProperties(_graph_handle, index, &arg);
        // if (ZE_RESULT_SUCCESS != result) {
        //     IE_THROW() << "Failed to get pfnGetArgumentProperties";
        // }

        IE::Precision net_precision = toIEPrecision(arg.networkPrecision);
        IE::Layout net_layout = toIELayout(arg.networkLayout);
        IE::SizeVector net_dims(arg.dims, arg.dims + getDimCount(net_layout));
        IE::TensorDesc net_dataDesc(net_precision, net_dims, net_layout);

        IE::Precision dev_precision = toIEPrecision(arg.devicePrecision);
        IE::Layout dev_layout = toIELayout(arg.deviceLayout);
        IE::SizeVector dev_dims(arg.dims, arg.dims + getDimCount(dev_layout));
        IE::TensorDesc dev_dataDesc(dev_precision, dev_dims, dev_layout);

        if (ZE_GRAPH_ARGUMENT_TYPE_INPUT == arg.type) {
            _logger.debug("Found input {}", arg.name);

            net_inputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, net_dataDesc));
            dev_inputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, dev_dataDesc));
        }

        // Same code
        if (ZE_GRAPH_ARGUMENT_TYPE_OUTPUT == arg.type) {
            _logger.debug("Found output {}", arg.name);

            net_outputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, net_dataDesc));
            dev_outputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, dev_dataDesc));
        }
    }

    return std::make_tuple(net_inputs, net_outputs, dev_inputs, dev_outputs);
}

}  // namespace zeroCompilerAdapter
}  // namespace vpux
