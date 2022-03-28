//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/device.hpp"
#include "vpux/IMD/executor.hpp"
#include "vpux/IMD/platform_helpers.hpp"

#include "vpux/al/config/common.hpp"

vpux::IMD::DeviceImpl::DeviceImpl(InferenceEngine::VPUXConfigParams::VPUXPlatform platform): _platform(platform) {
    VPUX_THROW_UNLESS(platformSupported(platform), "Unsupported VPUX platform '{0}'", platform);
}

std::shared_ptr<vpux::Allocator> vpux::IMD::DeviceImpl::getAllocator() const {
    return nullptr;
}

std::shared_ptr<vpux::Allocator> vpux::IMD::DeviceImpl::getAllocator(const InferenceEngine::ParamMap&) const {
    return nullptr;
}

std::shared_ptr<vpux::Executor> vpux::IMD::DeviceImpl::createExecutor(const NetworkDescription::Ptr& network,
                                                                      const Config& config) {
    return std::make_shared<vpux::IMD::ExecutorImpl>(_platform, network, config);
}

std::string vpux::IMD::DeviceImpl::getName() const {
    auto platformName = stringifyEnum(_platform);
    VPUX_THROW_UNLESS(platformName.consume_front("VPU"), "Unsupported VPUX platform '{0}'", _platform);
    return platformName.str();
}
