//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/device.hpp"
#include "vpux/IMD/executor.hpp"
#include "vpux/IMD/platform_helpers.hpp"

#include "vpux/al/config/common.hpp"

namespace vpux {

IMDDevice::IMDDevice(InferenceEngine::VPUXConfigParams::VPUXPlatform platform): _platform(platform) {
    VPUX_THROW_UNLESS(platformSupported(platform), "Unsupported VPUX platform '{0}'", stringifyEnum(platform));
}

std::shared_ptr<Executor> IMDDevice::createExecutor(const NetworkDescription::CPtr network, const Config& config) {
    return std::make_shared<IMDExecutor>(_platform, network, config);
}

std::string IMDDevice::getName() const {
    std::string_view platformName = stringifyEnum(_platform);
    static const std::string_view vpu_prefix("VPU");
    auto prefix_pos = platformName.find(vpu_prefix);
    VPUX_THROW_UNLESS(prefix_pos != platformName.npos, "Unsupported VPUX platform '{0}'",
                      static_cast<std::underlying_type_t<InferenceEngine::VPUXConfigParams::VPUXPlatform>>(_platform));
    platformName.remove_prefix(vpu_prefix.size());
    return platformName.data();
}

std::string IMDDevice::getFullDeviceName() const {
    return "Intel(R) NPU (IMD)";
}

}  // namespace vpux
