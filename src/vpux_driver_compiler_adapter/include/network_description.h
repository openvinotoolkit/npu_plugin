//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux_compiler.hpp"

namespace vpux {
namespace driverCompilerAdapter {

struct NetworkMeta final {
    NetworkIOVector deviceInputs;
    NetworkIOVector deviceOutputs;
    std::vector<OVRawNode> ovResults;
    std::vector<OVRawNode> ovParameters;
    int numStreams = 1;
};

/**
 * @brief Network blob + meta information
 */
class NetworkDescription final : public INetworkDescription {
public:
    NetworkDescription(const std::vector<char>& compiledNetwork, const std::string& name,
                       const NetworkIOVector& deviceInputs, const NetworkIOVector& deviceOutputs,
                       const std::vector<OVRawNode>& ovResults, const std::vector<OVRawNode>& ovParameters,
                       int numStreams)
            : _compiledNetwork(compiledNetwork),
              _name(name),
              _deviceInputs(deviceInputs),
              _deviceOutputs(deviceOutputs),
              _ovResults(ovResults),
              _ovParameters(ovParameters),
              _numStreams(numStreams) {
    }

    NetworkDescription(const std::vector<char>& compiledNetwork, const std::string& name,
                       const NetworkMeta& networkMeta)
            : NetworkDescription(compiledNetwork, name, networkMeta.deviceInputs, networkMeta.deviceOutputs,
                                 networkMeta.ovResults, networkMeta.ovParameters, networkMeta.numStreams) {
    }

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

    NetworkIOVector _deviceInputs;
    NetworkIOVector _deviceOutputs;

    // TODO #-30194 Add profiling support for driver compiler adapter and CID
    NetworkIOVector _profilingOutputs;

    // TODO #-30196 Support ovParameters and ovResults which required for OV2.0 support
    const std::vector<vpux::OVRawNode> _ovParameters;
    const std::vector<vpux::OVRawNode> _ovResults;

    int _numStreams = 1;
};

}  // namespace driverCompilerAdapter
}  // namespace vpux
