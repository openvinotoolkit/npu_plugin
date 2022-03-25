//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/platform_helpers.hpp"
#include "vpux/utils/core/enums.hpp"

namespace {
static const vpux::EnumMap<InferenceEngine::VPUXConfigParams::VPUXPlatform, std::string> supportedPlatforms = {
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720, "3720"},
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700, "ma2490"},
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400, "ma2490"}};
}

namespace vpux {
namespace IMD {

bool platformSupported(InferenceEngine::VPUXConfigParams::VPUXPlatform platform) {
    return supportedPlatforms.find(platform) != supportedPlatforms.end();
}

std::string getChipsetName(InferenceEngine::VPUXConfigParams::VPUXPlatform platform) {
    const auto plat_it = supportedPlatforms.find(platform);
    return plat_it != supportedPlatforms.end() ? plat_it->second : std::string{};
}

}  // namespace IMD
}  // namespace vpux
