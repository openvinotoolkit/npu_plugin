//
// Copyright 2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "device_helpers.hpp"

namespace ie = InferenceEngine;

const static std::map<uint32_t, ie::VPUXConfigParams::VPUXPlatform> platformIdMap = {
        {0, ie::VPUXConfigParams::VPUXPlatform::VPU3400_A0},  // KMB A0 / B0
        {1, ie::VPUXConfigParams::VPUXPlatform::VPU3800},     // TBH prime
        {2, ie::VPUXConfigParams::VPUXPlatform::VPU3900},     // TBH full
        {3, ie::VPUXConfigParams::VPUXPlatform::VPU3720},     // MTL
};

const static std::map<ie::VPUXConfigParams::VPUXPlatform, std::string> platformNameMap = {
        {ie::VPUXConfigParams::VPUXPlatform::AUTO, VPUX_CONFIG_VALUE(AUTO)},              // auto detection
        {ie::VPUXConfigParams::VPUXPlatform::VPU3400_A0, VPUX_CONFIG_VALUE(VPU3400_A0)},  // KMB A0
        {ie::VPUXConfigParams::VPUXPlatform::VPU3400, VPUX_CONFIG_VALUE(VPU3400)},        // KMB B0 400 MHz
        {ie::VPUXConfigParams::VPUXPlatform::VPU3700, VPUX_CONFIG_VALUE(VPU3700)},        // KMB B0 700 MHz
        {ie::VPUXConfigParams::VPUXPlatform::VPU3800, VPUX_CONFIG_VALUE(VPU3800)},        // TBH Prime
        {ie::VPUXConfigParams::VPUXPlatform::VPU3900, VPUX_CONFIG_VALUE(VPU3900)},        // TBH Full
        {ie::VPUXConfigParams::VPUXPlatform::VPU3720, VPUX_CONFIG_VALUE(VPU3720)},        // MTL
};

const static std::map<std::string, ie::VPUXConfigParams::VPUXPlatform> platformNameInverseMap = {
        {VPUX_CONFIG_VALUE(AUTO), ie::VPUXConfigParams::VPUXPlatform::AUTO},              // auto detection
        {VPUX_CONFIG_VALUE(VPU3400_A0), ie::VPUXConfigParams::VPUXPlatform::VPU3400_A0},  // KMB A0
        {VPUX_CONFIG_VALUE(VPU3400), ie::VPUXConfigParams::VPUXPlatform::VPU3400},        // KMB B0 400 MHz
        {VPUX_CONFIG_VALUE(VPU3700), ie::VPUXConfigParams::VPUXPlatform::VPU3700},        // KMB B0 700 MHz
        {VPUX_CONFIG_VALUE(VPU3800), ie::VPUXConfigParams::VPUXPlatform::VPU3800},        // TBH Prime
        {VPUX_CONFIG_VALUE(VPU3900), ie::VPUXConfigParams::VPUXPlatform::VPU3900},        // TBH Full
        {VPUX_CONFIG_VALUE(VPU3720), ie::VPUXConfigParams::VPUXPlatform::VPU3720},        // MTL
};

uint32_t utils::getSliceIdBySwDeviceId(const uint32_t swDevId) {
    // bits 3-1 define slice ID
    // right shift to omit bit 0, thus slice id is stored in bits 2-0
    // apply b111 mask to discard anything but slice ID
    uint32_t sliceId = (swDevId >> 1) & 0x7;
    return sliceId;
}

ie::VPUXConfigParams::VPUXPlatform utils::getPlatformBySwDeviceId(const uint32_t swDevId) {
    // bits 7-4 define platform
    // right shift to omit bits 0-3. after that platform code is stored in bits 3-0
    // apply b1111 mask to discard anything but platform code
    uint32_t platformId = (swDevId >> 4) & 0xf;
    return platformIdMap.at(platformId);
}

std::string utils::getDeviceNameBySwDeviceId(const uint32_t swDevId) {
    ie::VPUXConfigParams::VPUXPlatform platform = getPlatformBySwDeviceId(swDevId);
    uint32_t sliceId = getSliceIdBySwDeviceId(swDevId);
    std::string deviceName = platformNameMap.at(platform) + "." + std::to_string(sliceId);
    return deviceName;
}

std::string utils::getPlatformNameByDeviceName(const std::string& deviceName) {
    const auto platformPos = deviceName.find('.');
    if (platformPos == std::string::npos) {
        return deviceName;
    }

    return deviceName.substr(0, platformPos);
}

ie::VPUXConfigParams::VPUXPlatform utils::getPlatformByDeviceName(const std::string& deviceName) {
    const auto platformName = getPlatformNameByDeviceName(deviceName);
    return platformNameInverseMap.at(platformName);
}
