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

#include "zero_api_adapter.h"
#include "ie_layouts.h"
#include <chrono>

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
    SerializedIR serializeIR(std::vector<char>& xml, std::vector<char>& weights) {
        const uint32_t numberOfInputData = 2;
        const uint32_t xmlSize = static_cast<uint32_t>(xml.size());
        const uint32_t weightsSize = static_cast<uint32_t>(weights.size());

        const size_t sizeOfSerializedIR =
                sizeof(numberOfInputData) + sizeof(xmlSize) + xml.size() + sizeof(weightsSize) + weights.size();

        std::vector<uint8_t> serializedIR;
        serializedIR.resize(sizeOfSerializedIR);

        uint32_t offset = 0;
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
} // namespace 

// TODO Not all Precision from IE listed in ze_graph_ext
IE::Precision toIEPrecision(const ze_graph_argument_precision_t ze_precision) {
    switch( ze_precision ) {
        case ZE_GRAPH_ARGUMENT_PRECISION_FP32: return IE::Precision::FP32;
        case ZE_GRAPH_ARGUMENT_PRECISION_FP16: return IE::Precision::FP16;
        case ZE_GRAPH_ARGUMENT_PRECISION_UINT16: return IE::Precision::U16;
        case ZE_GRAPH_ARGUMENT_PRECISION_UINT8: return IE::Precision::U8;
        case ZE_GRAPH_ARGUMENT_PRECISION_INT32: return IE::Precision::I32;
        case ZE_GRAPH_ARGUMENT_PRECISION_INT16: return IE::Precision::I16;
        case ZE_GRAPH_ARGUMENT_PRECISION_INT8: return IE::Precision::I8;
        case ZE_GRAPH_ARGUMENT_PRECISION_BIN: return IE::Precision::BIN;

        default: return IE::Precision::UNSPECIFIED;
    }
}

IE::Layout toIELayout(const ze_graph_argument_layout_t ze_layout) {
    switch (ze_layout) {
        case ZE_GRAPH_ARGUMENT_LAYOUT_NCHW: return IE::Layout::NCHW;
        case ZE_GRAPH_ARGUMENT_LAYOUT_NHWC: return IE::Layout::NHWC;
        case ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW: return IE::Layout::NCDHW;
        case ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC: return IE::Layout::NDHWC;

        case ZE_GRAPH_ARGUMENT_LAYOUT_OIHW: return IE::Layout::OIHW;
        
        case ZE_GRAPH_ARGUMENT_LAYOUT_C: return IE::Layout::C;

        case ZE_GRAPH_ARGUMENT_LAYOUT_CHW: return IE::Layout::CHW;

        case ZE_GRAPH_ARGUMENT_LAYOUT_HW: return IE::Layout::HW;
        case ZE_GRAPH_ARGUMENT_LAYOUT_NC: return IE::Layout::NC;
        case ZE_GRAPH_ARGUMENT_LAYOUT_CN: return IE::Layout::CN;

        case ZE_GRAPH_ARGUMENT_LAYOUT_BLOCKED: return IE::Layout::BLOCKED;
        default: return IE::Layout::ANY;
    }
}

size_t getDimCount( const IE::Layout layout )
{
    switch( layout ) {
        case IE::Layout::C: return 1;
        case IE::Layout::CN: return 2;
        case IE::Layout::HW: return 2;
        case IE::Layout::NC: return 2;
        case IE::Layout::CHW: return 3;
        case IE::Layout::NCHW: return 4;
        case IE::Layout::NHWC: return 4;
        case IE::Layout::NCDHW: return 5;
        case IE::Layout::NDHWC: return 5;
    }

    return 0;
}

//------------------------------------------------------------------------------
ZeroAPICompilerInDriver::ZeroAPICompilerInDriver() {
    _logger->debug("ZeroAPICompilerInDriver::ZeroAPICompilerInDriver");
    auto result = zeInit(0);
    if (ZE_RESULT_SUCCESS != result) {
        std::cerr << "ZeroDevicesSingleton zeInit failed 0x" << std::hex << uint64_t(result) << std::dec
                    << std::endl;
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

    _logger->debug("ZeroAPICompilerInDriver::ZeroAPICompilerInDriver done");
}

ZeroAPICompilerInDriver::~ZeroAPICompilerInDriver() {
    _logger->debug("ZeroAPICompilerInDriver::~ZeroAPICompilerInDriver");
    _logger->error("ZeroAPICompilerInDriver::~ZeroAPICompilerInDriver not implemented");
}

Blob::Ptr ZeroAPICompilerInDriver::compileIR(std::vector<char>& xml, std::vector<char>& weights) {
    using ms = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();
    _logger->debug("ZeroAPICompilerInDriver::compileIR");
    auto serializedIR = serializeIR(xml, weights);

    _logger->debug("serializedIR.size() {}", serializedIR.size());
    _logger->debug("serializedIR.data() {}", serializedIR.data());
    ze_graph_format_t format = ZE_GRAPH_FORMAT_NGRAPH_LITE;
    ze_graph_desc_t desc = { format, serializedIR.size(), serializedIR.data() };

    auto result = _graph_ddi_table_ext->pfnCreate( _device_handle, &desc, &_graph_handle);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to compile graph with zero API";
    }

    size_t blobSize = -1;
    result = _graph_ddi_table_ext->pfnGetNativeBinary( _graph_handle, &blobSize, nullptr);
    // uint8_t* blobMemotyPtr = new uint8_t[blobSize];
    // memset(blobMemotyPtr, 0, blobSize);
    
    std::vector<char> blob;
    blob.resize(blobSize);
    result = _graph_ddi_table_ext->pfnGetNativeBinary( _graph_handle, &blobSize,  reinterpret_cast<uint8_t*>(blob.data()));
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get native binary";
    }

    ze_context_desc_t context_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, 0, 0};
    result = zeContextCreate(_driver_handle, &context_desc, &_context);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to create context";
    }

    _logger->debug("Blob size = {}", blobSize);
    // _logger->debug("blobMemotyPtr == nullptr? {}", blobMemotyPtr == nullptr);

    // std::vector<char> blob;
    // blob.resize(blobSize);
    
    // FIXME additional memory copy operation which we can get rid of?
    // ie_memcpy(blob.data(), blob.size(), blobMemotyPtr, blobSize);

    _logger->debug("ZeroAPICompilerInDriver::compileIR end");
    auto finish = std::chrono::high_resolution_clock::now();
    _logger->info("|| Timer ||;ZeroAPICompilerInDriver::compileIR (ms);\t{}", std::chrono::duration_cast<ms>(finish - start).count());
    return std::make_shared<Blob>(blob);
}

void* ZeroAPICompilerInDriver::compileIRReturnHandle(std::vector<char>& xml, std::vector<char>& weights) {
    _logger->debug("ZeroAPICompilerInDriver::compileIRReturnHandle");
    auto serializedIR = serializeIR(xml, weights);

    _logger->debug("serializedIR.size() {}", serializedIR.size());
    _logger->debug("serializedIR.data() {}", serializedIR.data());
    ze_graph_format_t format = ZE_GRAPH_FORMAT_NGRAPH_LITE;
    ze_graph_desc_t desc = { format, serializedIR.size(), serializedIR.data() };

    auto result = _graph_ddi_table_ext->pfnCreate( _device_handle, &desc, &_graph_handle);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to compile graph with zero API";
    }
    _logger->debug("compileIRReturnHandle handle: {}", _graph_handle);
    _logger->debug("size of graph handle {}", sizeof(_graph_handle));
    return _graph_handle;
}

// ZeroAPICompilerInDriver::getNetworkMeta(void* graph_handle) {
std::tuple<const std::string, const DataMap, const DataMap, const DataMap, const DataMap>
ZeroAPICompilerInDriver::getNetworkMeta(const std::vector<char>& blob) {
    using ms = std::chrono::milliseconds;
    auto start = std::chrono::high_resolution_clock::now();
    _logger->debug("ZeroAPICompilerInDriver::getNetworkMeta");

    if (blob.size() > 0) {
        _logger->debug("Import network case");
        ze_graph_format_t format = ZE_GRAPH_FORMAT_NATIVE;
        ze_graph_desc_t desc = { format, blob.size(), reinterpret_cast<const uint8_t*>(blob.data()) };

        auto result = _graph_ddi_table_ext->pfnCreate( _device_handle, &desc, &_graph_handle);
        if (ZE_RESULT_SUCCESS != result) {
            IE_THROW() << "Failed to load blob with zero API";
        }
        _logger->debug("compileIRReturnHandle handle: {}", _graph_handle);
        _logger->debug("size of graph handle {}", sizeof(_graph_handle));
    }

    ze_graph_properties_t graph_properties{};
    auto result = _graph_ddi_table_ext->pfnGetProperties(_graph_handle, &graph_properties);
    if (ZE_RESULT_SUCCESS != result) {
        IE_THROW() << "Failed to get pfnGetProperties";
    }

    DataMap net_inputs;
    DataMap dev_inputs;

    DataMap net_outputs;
    DataMap dev_outputs;
    
    for (uint32_t index = 0; index < graph_properties.numGraphArgs; ++index) {
        ze_graph_argument_properties2_t arg;
        result = _graph_ddi_table_ext->pfnGetArgumentProperties2(_graph_handle, index, &arg);
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
            _logger->debug("Found input {}", arg.name);

            net_inputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, net_dataDesc));
            dev_inputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, dev_dataDesc));
        }

        // Same code
        if (ZE_GRAPH_ARGUMENT_TYPE_OUTPUT == arg.type) {
            _logger->debug("Found output {}", arg.name);

            net_outputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, net_dataDesc));
            dev_outputs.emplace(arg.name, std::make_shared<IE::Data>(arg.name, dev_dataDesc));
        }
    }

    // TODO can we get name some way?
    const std::string graphName = "graph";
    // std::cout << "_graph_handle " << _graph_handle << std::endl;
    // std::cout << "_device_handle " << _device_handle << std::endl;
    // std::cout << "_driver_handle " << _driver_handle << std::endl;

    _logger->debug("ZeroAPICompilerInDriver::getDeviceNetworkMeta done");
    auto finish = std::chrono::high_resolution_clock::now();
    _logger->info("|| Timer ||;ZeroAPICompilerInDriver::getNetworkMeta (ms);\t{}", std::chrono::duration_cast<ms>(finish - start).count());
    return std::make_tuple(graphName, net_inputs, net_outputs, dev_inputs, dev_outputs);
}

std::tuple<const DataMap, const DataMap> ZeroAPICompilerInDriver::getDeviceNetworkMeta(const Blob::Ptr compiledNetwork) {
    _logger->debug("ZeroAPICompilerInDriver::getDeviceNetworkMeta");
    _logger->error("ZeroAPICompilerInDriver::getDeviceNetworkMeta not implemented");
    return std::make_tuple(DataMap(), DataMap());
}

Opset ZeroAPICompilerInDriver::getSupportedOpset() {
    _logger->debug("ZeroAPICompilerInDriver::getSupportedOpset");
    _logger->error("ZeroAPICompilerInDriver::getSupportedOpset not implemented");
    return {0}; 
}

}  // namespace zeroCompilerAdapter
}  // namespace vpux