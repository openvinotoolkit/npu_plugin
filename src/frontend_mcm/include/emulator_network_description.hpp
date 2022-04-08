//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"
#include "vpux_compiler.hpp"

#include <include/mcm/compiler/compilation_unit.hpp>

#include <memory>

namespace vpu {
namespace MCMAdapter {

class EmulatorNetworkDescription final : public vpux::INetworkDescription {
public:
    EmulatorNetworkDescription(const std::vector<char>& compiledNetwork, const vpux::Config& config,
                               const std::string& name);

    const vpux::DataMap& getInputsInfo() const final;

    const vpux::DataMap& getOutputsInfo() const final;

    const vpux::DataMap& getDeviceInputsInfo() const final;

    const vpux::DataMap& getDeviceOutputsInfo() const final;

    const vpux::DataMap& getDeviceProfilingOutputsInfo() const final;

    const std::vector<vpux::OVRawNode>& getOVParameters() const final;

    const std::vector<vpux::OVRawNode>& getOVResults() const final;

    const vpux::QuantizationParamMap& getQuantParamsInfo() const final;

    const std::vector<char>& getCompiledNetwork() const final;

    const void* getNetworkModel() const final;

    std::size_t getNetworkModelSize() const final;  // not relevant information for this type

    const std::string& getName() const final;

    int getNumStreams() const final {
        return 1;
    }

private:
    std::string _name;
    vpux::Logger _logger;
    vpux::DataMap _dataMapPlaceholder;
    vpux::DataMap _deviceInputs;
    vpux::DataMap _deviceOutputs;
    vpux::DataMap _networkInputs;
    vpux::DataMap _networkOutputs;
    std::vector<char> _compiledNetwork;
    vpux::QuantizationParamMap _quantParams;
    const std::vector<vpux::OVRawNode> _ovParameters;
    const std::vector<vpux::OVRawNode> _ovResults;
};

}  // namespace MCMAdapter
}  // namespace vpu
