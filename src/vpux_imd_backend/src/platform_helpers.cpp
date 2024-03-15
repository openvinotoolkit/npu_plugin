//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/IMD/platform_helpers.hpp"

#include "vpux/al/config/common.hpp"

#include "vpux/utils/core/enums.hpp"
#include "vpux/utils/core/error.hpp"

using InferenceEngine::VPUXConfigParams::VPUXPlatform;

namespace vpux {

namespace {

const EnumMap<VPUXPlatform, StringRef> platformToAppNameMap = {
        {VPUXPlatform::VPU3700, "InferenceManagerDemo_vpu_2_0.elf"},
        {VPUXPlatform::VPU3720, "InferenceManagerDemo_vpu_2_7.elf"},
};

}  // namespace

bool platformSupported(VPUXPlatform platform) {
    return platformToAppNameMap.find(platform) != platformToAppNameMap.end();
}

StringRef getAppName(VPUXPlatform platform) {
    const auto it = platformToAppNameMap.find(platform);
    VPUX_THROW_WHEN(it == platformToAppNameMap.end(), "Platform '{0}' is not supported", stringifyEnum(platform));
    return it->second;
}

}  // namespace vpux
