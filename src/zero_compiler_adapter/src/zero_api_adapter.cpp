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

    // Device part removed
    _logger->debug("ZeroAPICompilerInDriver::ZeroAPICompilerInDriver done");
}

ZeroAPICompilerInDriver::~ZeroAPICompilerInDriver() {
    _logger->debug("ZeroAPICompilerInDriver::~ZeroAPICompilerInDriver");
    _logger->error("ZeroAPICompilerInDriver::~ZeroAPICompilerInDriver not implemented");
}

Blob::Ptr ZeroAPICompilerInDriver::compileIR(std::vector<char>& xml, std::vector<char>& weights) {
    _logger->debug("ZeroAPICompilerInDriver::compileIR");
    _logger->error("ZeroAPICompilerInDriver::compileIR not implemented");
    std::vector<char> blob;
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