//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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

    Executor::Ptr createExecutor(const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) override;

private:
    std::shared_ptr<Allocator> _allocatorPtr = nullptr;
    const std::string _name;
};

}  // namespace hddl2
}  // namespace vpux
