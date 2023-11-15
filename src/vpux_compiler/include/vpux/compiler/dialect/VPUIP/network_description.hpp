//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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

    const NetworkIOVector& getDeviceInputsInfo() const final {
        return _deviceInputs;
    }
    const NetworkIOVector& getDeviceOutputsInfo() const final {
        return _deviceOutputs;
    }
    const NetworkIOVector& getDeviceProfilingOutputsInfo() const final {
        return _deviceProfilingOutputs;
    }

    const std::vector<OVRawNode>& getOVParameters() const final {
        return _ovParameters;
    }

    const std::vector<OVRawNode>& getOVResults() const final {
        return _ovResults;
    }

    int getNumStreams() const final {
        return _numStreams;
    }

private:
    std::vector<char> _compiledNetwork;

    std::string _name;

    NetworkIOVector _deviceInputs;
    NetworkIOVector _deviceOutputs;
    NetworkIOVector _deviceProfilingOutputs;

    std::vector<OVRawNode> _ovResults;
    std::vector<OVRawNode> _ovParameters;

    int _numStreams = 1;
};

}  // namespace VPUIP
}  // namespace vpux
