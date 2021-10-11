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
    _logger->error("ZeroAPICompilerInDriver::ZeroAPICompilerInDriver not implemented");
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