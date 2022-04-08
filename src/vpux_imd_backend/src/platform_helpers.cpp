//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/platform_helpers.hpp"

#include "vpux/al/config/common.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"

using namespace vpux;
using InferenceEngine::VPUXConfigParams::VPUXPlatform;

namespace {

const EnumMap<VPUXPlatform, StringRef> platformToAppNameMap = {
        {VPUXPlatform::VPU3720, "InferenceManagerDemo_vpu_2_7.elf"},  //
        {VPUXPlatform::VPU3700, "InferenceManagerDemo_vpu_2_0.elf"},  //
        {VPUXPlatform::VPU3400, "InferenceManagerDemo_vpu_2_0.elf"},  //
};

}  // namespace

bool vpux::IMD::platformSupported(VPUXPlatform platform) {
    return platformToAppNameMap.find(platform) != platformToAppNameMap.end();
}

StringRef vpux::IMD::getAppName(VPUXPlatform platform) {
    const auto it = platformToAppNameMap.find(platform);
    VPUX_THROW_WHEN(it == platformToAppNameMap.end(), "Platform '{0}' is not supported", platform);
    return it->second;
}
