//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux_compiler.hpp"

namespace vpux {
namespace driverCompilerAdapter {

struct NetworkMeta final {
    DataMap networkInputs;
    DataMap networkOutputs;
    DataMap deviceInputs;
    DataMap deviceOutputs;
    std::vector<OVRawNode> ovResults;
    std::vector<OVRawNode> ovParameters;
    int numStreams = 1;
};

/**
 * @brief Network blob + meta information
 */
class NetworkDescription final : public INetworkDescription {
public:
    NetworkDescription(const std::vector<char>& compiledNetwork, const std::string& name, const DataMap& networkInputs,
                       const DataMap& networkOutputs, const DataMap& deviceInputs, const DataMap& deviceOutputs,
                       const std::vector<OVRawNode>& ovResults, const std::vector<OVRawNode>& ovParameters,
                       int numStreams)
            : _compiledNetwork(compiledNetwork),
              _name(name),
              _networkInputs(networkInputs),
              _networkOutputs(networkOutputs),
              _deviceInputs(deviceInputs),
              _deviceOutputs(deviceOutputs),
              _ovResults(ovResults),
              _ovParameters(ovParameters),
              _numStreams(numStreams) {
    }

    NetworkDescription(const std::vector<char>& compiledNetwork, const std::string& name,
                       const NetworkMeta& networkMeta)
            : NetworkDescription(compiledNetwork, name, networkMeta.networkInputs, networkMeta.networkOutputs,
                                 networkMeta.deviceInputs, networkMeta.deviceOutputs, networkMeta.ovResults,
                                 networkMeta.ovParameters, networkMeta.numStreams) {
    }

public:
    const vpux::QuantizationParamMap& getQuantParamsInfo() const final {
        return _quantParams;
    }

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
    const DataMap& getDeviceProfilingOutputsInfo() const final {
        return _profilingOutputs;
    }

    const std::vector<vpux::OVRawNode>& getOVParameters() const final {
        return _ovParameters;
    }
    const std::vector<vpux::OVRawNode>& getOVResults() const final {
        return _ovResults;
    }

    int getNumStreams() const final {
        return _numStreams;
    }

private:
    std::vector<char> _compiledNetwork;

    std::string _name;

    DataMap _networkInputs;
    DataMap _networkOutputs;

    DataMap _deviceInputs;
    DataMap _deviceOutputs;

    // TODO #-30194 Add profiling support for driver compiler adapter and CID
    DataMap _profilingOutputs;

    // TODO #-30195 Add quant params support
    vpux::QuantizationParamMap _quantParams{};

    // TODO #-30196 Support ovParameters and ovResults which required for OV2.0 support
    const std::vector<vpux::OVRawNode> _ovParameters;
    const std::vector<vpux::OVRawNode> _ovResults;

    int _numStreams = 1;
};

}  // namespace driverCompilerAdapter
}  // namespace vpux
