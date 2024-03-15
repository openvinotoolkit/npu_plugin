//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux_compiler.hpp"

namespace vpux {
namespace driverCompilerAdapter {

struct NetworkMeta final {
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<std::string> stateNames;

    IONodeDescriptorMap parameters;
    IONodeDescriptorMap results;
    IONodeDescriptorMap states;

    int numStreams = 1;
};

/**
 * @brief Network blob + meta information
 */
class NetworkDescription final : public INetworkDescription {
public:
    NetworkDescription(const std::vector<char>& compiledNetwork, const std::string& name,
                       const std::vector<std::string>& inputNames, const std::vector<std::string>& outputNames,
                       const std::vector<std::string>& stateNames, const IONodeDescriptorMap& parameters,
                       const IONodeDescriptorMap& results, const IONodeDescriptorMap& states, int numStreams) {
        _compiledNetwork = compiledNetwork;
        _name = name;
        _inputNames = inputNames;
        _outputNames = outputNames;
        _stateNames = stateNames;
        _parameters = parameters;
        _results = results;
        _states = states;
        _numStreams = numStreams;
    }

    NetworkDescription(const std::vector<char>& compiledNetwork, const std::string& name,
                       const NetworkMeta& networkMeta)
            : NetworkDescription(compiledNetwork, name, networkMeta.inputNames, networkMeta.outputNames,
                                 networkMeta.stateNames, networkMeta.parameters, networkMeta.results,
                                 networkMeta.states, networkMeta.numStreams) {
    }
};

}  // namespace driverCompilerAdapter
}  // namespace vpux
