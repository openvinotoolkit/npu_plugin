//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "infer_request.hpp"
#include "vpux.hpp"
#include "vpux_private_config.hpp"

namespace vpux {
namespace IMD {

class DeviceImpl final : public IDevice {
public:
    explicit DeviceImpl(InferenceEngine::VPUXConfigParams::VPUXPlatform platform);

public:
    std::shared_ptr<Allocator> getAllocator() const override;
    std::shared_ptr<Allocator> getAllocator(const InferenceEngine::ParamMap& params) const override;

    std::shared_ptr<Executor> createExecutor(const NetworkDescription::Ptr& network, const Config& config) override;

    std::string getName() const override;
    std::string getFullDeviceName() const override;

    IInferRequest::Ptr createInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                                          const InferenceEngine::OutputsDataMap& networkOutputs,
                                          const Executor::Ptr& executor, const Config& config,
                                          const std::string& networkName,
                                          const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                          const std::vector<std::shared_ptr<const ov::Node>>& results,
                                          const vpux::NetworkIOVector& networkStatesInfo,
                                          const std::shared_ptr<InferenceEngine::IAllocator>& allocator) override {
        return std::make_shared<IMDInferRequest>(networkInputs, networkOutputs, executor, config, networkName,
                                                 parameters, results, networkStatesInfo, allocator);
    }

private:
    InferenceEngine::VPUXConfigParams::VPUXPlatform _platform;
};

}  // namespace IMD
}  // namespace vpux
