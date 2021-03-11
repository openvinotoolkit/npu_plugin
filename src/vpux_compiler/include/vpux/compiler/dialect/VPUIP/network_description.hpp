//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include "vpux_compiler.hpp"

namespace vpux {
namespace VPUIP {

class NetworkDescription final : public INetworkDescription {
public:
    explicit NetworkDescription(std::vector<char> blob);

public:
    const std::vector<char>& getCompiledNetwork() const final {
        return _compiledNetwork;
    }

    const void* getNetworkModel() const final {
        return _compiledNetwork.data();
    }

    std::size_t getNetworkModelSize() const final {
        return _compiledNetwork.size();
    }

    const std::string& getName() const final {
        return _name;
    }

    const DataMap& getInputsInfo() const final {
        return _networkInputs;
    }
    const DataMap& getOutputsInfo() const final {
        return _networkOutputs;
    }

    const DataMap& getDeviceInputsInfo() const final {
        return _deviceInputs;
    }
    const DataMap& getDeviceOutputsInfo() const final {
        return _deviceOutputs;
    }

private:
    std::vector<char> _compiledNetwork;

    std::string _name;

    DataMap _networkInputs;
    DataMap _networkOutputs;

    DataMap _deviceInputs;
    DataMap _deviceOutputs;
};

}  // namespace VPUIP
}  // namespace vpux
