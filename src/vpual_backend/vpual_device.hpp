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

#include <ie_allocator.hpp>
#include <memory>
#include <string>
#include <vpual_config.hpp>
#include <vpux.hpp>

namespace vpux {

class VpualDevice final : public IDevice {
public:
    VpualDevice(const std::string& name, const InferenceEngine::VPUXConfigParams::VPUXPlatform& platform);
    std::shared_ptr<Allocator> getAllocator() const override;
    std::shared_ptr<Allocator> getAllocator(const InferenceEngine::ParamMap& paramMap) const override;

    std::shared_ptr<Executor> createExecutor(
        const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) override;

    std::string getName() const override;

private:
    std::shared_ptr<Allocator> _allocator;
    const std::string _name;
    const InferenceEngine::VPUXConfigParams::VPUXPlatform _platform;
    // TODO: config is used in executor only
    // it makes sense to store it only there
    VpualConfig _config;
};

}  // namespace vpux
