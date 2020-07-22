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

#include <vpux_compiler.hpp>
#include <vpu/utils/logger.hpp>

#include "mcm_config.h"

namespace vpu {
namespace MCMAdapter {

class MCMNetworkDescription final : public vpux::NetworkDescription {
public:
    MCMNetworkDescription(const std::vector<char>& compiledNetwork, const MCMConfig& config);
    const vpux::DataMap& getInputsInfo() const override;

    const vpux::DataMap& getOutputsInfo() const override;

    const vpux::DataMap& getDeviceInputsInfo() const override;

    const vpux::DataMap& getDeviceOutputsInfo() const override;

    const std::vector<char>& getCompiledNetwork() const override;

private:
    const std::vector<char> _compiledNetwork;
    vpux::DataMap _deviceInputs;
    vpux::DataMap _deviceOutputs;

    vpux::DataMap _networkInputs;
    vpux::DataMap _networkOutputs;

    std::shared_ptr<vpu::Logger> _logger;

    vpux::DataMap matchElementsByName(const vpux::DataMap& actualDeviceData,
                                      const std::vector<std::string>& names);
    vpux::DataMap matchElementsByLexicographicalOrder(const vpux::DataMap& actualDeviceData,
                                                      const std::vector<std::string>& names);
    vpux::DataMap createDeviceMapWithCorrectNames(const vpux::DataMap& actualDeviceData,
                                                  const std::vector<std::string>& names);
};

// TODO find a better place for this helpers
namespace helpers {
    InferenceEngine::InputsDataMap dataMapIntoInputsDataMap(const vpux::DataMap& dataMap);
    InferenceEngine::OutputsDataMap dataMapIntoOutputsDataMap(const vpux::DataMap& dataMap);
}   // namespace helpers

}  // namespace MCMAdapter
}  // namespace vpu
