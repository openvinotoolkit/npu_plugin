//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/core/logger.hpp"
#include "vpux_compiler.hpp"

namespace vpu {
namespace MCMAdapter {

class MCMNetworkDescription final : public vpux::INetworkDescription {
public:
    // TODO extract network name from blob
    MCMNetworkDescription(const std::vector<char>& compiledNetwork, const vpux::Config& config,
                          const std::string& name = "");
    const vpux::DataMap& getInputsInfo() const override;

    const vpux::DataMap& getOutputsInfo() const override;

    const vpux::DataMap& getDeviceInputsInfo() const override;

    const vpux::DataMap& getDeviceOutputsInfo() const override;

    const vpux::DataMap& getDeviceProfilingOutputsInfo() const override;

    const std::vector<vpux::OVRawNode>& getOVParameters() const override;

    const std::vector<vpux::OVRawNode>& getOVResults() const override;

    const vpux::QuantizationParamMap& getQuantParamsInfo() const override;

    const std::vector<char>& getCompiledNetwork() const override;

    const void* getNetworkModel() const override;

    std::size_t getNetworkModelSize() const override;

    const std::string& getName() const override;

    int getNumStreams() const final {
        return _numStreams;
    }

private:
    std::string _name;
    const std::vector<char> _compiledNetwork;

    vpux::DataMap _deviceInputs;
    vpux::DataMap _deviceOutputs;
    vpux::DataMap _deviceProfilingOutputs;

    vpux::DataMap _networkInputs;
    vpux::DataMap _networkOutputs;

    std::vector<vpux::OVRawNode> _ovResults;
    std::vector<vpux::OVRawNode> _ovParameters;

    vpux::QuantizationParamMap _quantParams;

    int _numStreams = 1;

    vpux::Logger _logger;

    vpux::DataMap matchElementsByName(const vpux::DataMap& actualDeviceData, const std::vector<std::string>& names);
    vpux::DataMap matchElementsByLexicographicalOrder(const vpux::DataMap& actualDeviceData,
                                                      const std::vector<std::string>& names);
    vpux::DataMap createDeviceMapWithCorrectNames(const vpux::DataMap& actualDeviceData,
                                                  const std::vector<std::string>& names);
};
}  // namespace MCMAdapter
}  // namespace vpu
