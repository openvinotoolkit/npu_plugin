//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux.hpp"
#include "vpux_private_config.hpp"

#include <ie_allocator.hpp>

#include <memory>
#include <string>

namespace vpux {

class VpualDevice final : public IDevice {
public:
    VpualDevice(const std::string& name, const InferenceEngine::VPUXConfigParams::VPUXPlatform& platform);
    std::shared_ptr<Allocator> getAllocator() const override;
    std::shared_ptr<Allocator> getAllocator(const InferenceEngine::ParamMap& paramMap) const override;

    std::shared_ptr<Executor> createExecutor(const NetworkDescription::Ptr& networkDescription,
                                             const Config& config) override;

    std::string getName() const override;

    // TODO: it is a stub for future implementation
    // currently, nullptr is used as a signal to use InferRequest from vpux_al
    InferRequest::Ptr createInferRequest(const InferenceEngine::InputsDataMap& /*networkInputs*/,
                                         const InferenceEngine::OutputsDataMap& /*networkOutputs*/,
                                         const Executor::Ptr& /*executor*/, const Config& /*config*/,
                                         const std::string& /*networkName*/,
                                         const std::vector<std::shared_ptr<const ov::Node>>& /*parameters*/,
                                         const std::vector<std::shared_ptr<const ov::Node>>& /*results*/,
                                         const std::shared_ptr<InferenceEngine::IAllocator>& /*allocator*/) override {
        return nullptr;
    }

private:
    std::shared_ptr<Allocator> _allocator;
    const std::string _name;
    const InferenceEngine::VPUXConfigParams::VPUXPlatform _platform;
};

}  // namespace vpux
