//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

// System
#include <memory>
#include <string>

// IE
#include "ie_allocator.hpp"

// Plugin
#include "vpux.hpp"

namespace vpux {
namespace hddl2 {

/**
 * @brief General device, for ImageWorkload.
 * If specific name not provided, device selection will be postponed until inference
 */
class ImageWorkloadDevice final : public IDevice {
public:
    explicit ImageWorkloadDevice(const std::string& name = "");
    std::shared_ptr<Allocator> getAllocator() const override {
        return nullptr;
    }
    // For generic device no specific allocator can be provided
    std::shared_ptr<Allocator> getAllocator(const InferenceEngine::ParamMap& paramMap) const override;

    std::string getName() const override {
        return _name;
    }

    Executor::Ptr createExecutor(const NetworkDescription::Ptr& networkDescription, const Config& config) override;

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
    std::shared_ptr<Allocator> _allocatorPtr = nullptr;
    const std::string _name;
};

}  // namespace hddl2
}  // namespace vpux
