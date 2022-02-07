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
