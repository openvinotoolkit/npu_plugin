//
// Copyright Intel Corporation.
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
