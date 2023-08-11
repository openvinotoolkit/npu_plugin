//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/compiler/dialect/ELF/metadata.hpp"
#include "vpux_compiler.hpp"

namespace vpux {
namespace VPUMI37XX {

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

    const QuantizationParamMap& getQuantParamsInfo() const final {
        return _quantParams;
    }

    const DataMap& getDeviceInputsInfo() const final {
        return _deviceInputs;
    }
    const DataMap& getDeviceOutputsInfo() const final {
        return _deviceOutputs;
    }
    const DataMap& getDeviceProfilingOutputsInfo() const final {
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

    std::string _name = "ELF_BLOB";

    DataMap _networkInputs;
    DataMap _networkOutputs;

    DataMap _deviceInputs;
    DataMap _deviceOutputs;
    DataMap _deviceProfilingOutputs;
    QuantizationParamMap _quantParams;

    std::vector<OVRawNode> _ovResults;
    std::vector<OVRawNode> _ovParameters;

    int _numStreams = 1;
};

}  // namespace VPUMI37XX
}  // namespace vpux
