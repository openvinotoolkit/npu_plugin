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

namespace vpux {
namespace zeroCompilerAdapter {


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
    _logger->debug("ZeroAPICompilerInDriver::compileIR");
    auto serializedIR = serializeIR(xml, weights);

    ze_graph_format_t format = ZE_GRAPH_FORMAT_NGRAPH_LITE;
    ze_graph_desc_t desc = { format, serializedIR.size(), serializedIR.data() };

    _graph_ddi_table_ext->pfnCreate( _device_handle, &desc, &_graph_handle);

    std::vector<char> blob;
    _logger->debug("ZeroAPICompilerInDriver::compileIR end");
    return std::make_shared<Blob>(blob);
}

std::tuple<const std::string, const DataMap, const DataMap, const DataMap, const DataMap>
ZeroAPICompilerInDriver::getNetworkMeta(const Blob::Ptr compiledNetwork) {
    _logger->debug("ZeroAPICompilerInDriver::getNetworkMeta");
    _logger->error("ZeroAPICompilerInDriver::getNetworkMeta not implemnented");
    return std::make_tuple(std::string(), DataMap(), DataMap(), DataMap(), DataMap());
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