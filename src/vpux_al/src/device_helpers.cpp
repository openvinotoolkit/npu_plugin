//
// Copyright 2021 Intel Corporation.
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

#include <device_helpers.hpp>
#include <iostream>

namespace ie = InferenceEngine;

const static std::map<uint32_t, ie::VPUXConfigParams::VPUXPlatform> platformIdMap = {
        {0, ie::VPUXConfigParams::VPUXPlatform::VPU3400_A0},  // KMB A0 / B0
        {1, ie::VPUXConfigParams::VPUXPlatform::VPU3800},     // TBH prime
        {2, ie::VPUXConfigParams::VPUXPlatform::VPU3900},     // TBH full
        {3, ie::VPUXConfigParams::VPUXPlatform::VPU3720},     // MTL
};

const static std::map<ie::VPUXConfigParams::VPUXPlatform, std::string> platformNameMap = {
        {ie::VPUXConfigParams::VPUXPlatform::AUTO, "AUTO"},           // auto detection
        {ie::VPUXConfigParams::VPUXPlatform::VPU3400_A0, "3400_A0"},  // KMB A0
        {ie::VPUXConfigParams::VPUXPlatform::VPU3400, "3400"},        // KMB B0 400 MHz
        {ie::VPUXConfigParams::VPUXPlatform::VPU3700, "3700"},        // KMB B0 700 MHz
        {ie::VPUXConfigParams::VPUXPlatform::VPU3800, "3800"},        // TBH Prime
        {ie::VPUXConfigParams::VPUXPlatform::VPU3900, "3900"},        // TBH Full
        {ie::VPUXConfigParams::VPUXPlatform::VPU3720, "3720"},        // MTL
};

const static std::map<std::string, ie::VPUXConfigParams::VPUXPlatform> platformNameInverseMap = {
        {"AUTO", ie::VPUXConfigParams::VPUXPlatform::AUTO},           // auto detection
        {"3400_A0", ie::VPUXConfigParams::VPUXPlatform::VPU3400_A0},  // KMB A0
        {"3400", ie::VPUXConfigParams::VPUXPlatform::VPU3400},        // KMB B0 400 MHz
        {"3700", ie::VPUXConfigParams::VPUXPlatform::VPU3700},        // KMB B0 700 MHz
        {"3800", ie::VPUXConfigParams::VPUXPlatform::VPU3800},        // TBH Prime
        {"3900", ie::VPUXConfigParams::VPUXPlatform::VPU3900},        // TBH Full
        {"3720", ie::VPUXConfigParams::VPUXPlatform::VPU3720},        // MTL
};

bool utils::isVPUDevice(const uint32_t deviceId) {
    // bits 26-24 define interface type
    // 000 - IPC
    // 001 - PCIe
    // 010 - USB
    // 011 - ethernet
    constexpr uint32_t INTERFACE_TYPE_SELECTOR = 0x7000000;
    uint32_t interfaceType = (deviceId & INTERFACE_TYPE_SELECTOR);
    return (interfaceType == 0);
}

uint32_t utils::getSliceIdBySwDeviceId(const uint32_t swDevId) {
    // bits 3-1 define slice ID
    // right shift to omit bit 0, thus slice id is stored in bits 2-0
    // apply b111 mask to discard anything but slice ID
    uint32_t sliceId = (swDevId >> 1) & 0x7;
    return sliceId;
}

int utils::getSliceIdByDeviceName(const std::string& deviceName) {
    // Empty device name - return the first slice
    if (deviceName.empty()) {
        return 0;
    }

    // Check "only platform" naming format. For it return the first slice as well
    const auto platformName = utils::getPlatformNameByDeviceName(deviceName);
    if (platformName == deviceName) {
        return 0;
    }

    // Check "platform.slice_id" naming format
    int sliceId = 0;
    const int minSliceId = 0;
    const int maxSliceId = 3;
    const auto slicePos = deviceName.rfind('.');
    if (slicePos != std::string::npos) {
        const auto sliceStr = deviceName.substr(slicePos + 1, deviceName.length() - slicePos - 1);
        try {
            sliceId = std::stoi(sliceStr);
        } catch (const std::exception& ex) {
            IE_THROW() << "Device name conversion error - " << ex.what();
        }
        if ((sliceId < minSliceId) || (sliceId > maxSliceId)) {
            IE_THROW() << "Device name conversion error - bad slice number: " << sliceId;
        }
        return sliceId;
    }

    // TODO Remove this part after removing deprecated device names in future releases
    // Check deprecated "VPU-slice_id" naming format
    // *********************************************************************************
    for (sliceId = minSliceId; sliceId <= maxSliceId; ++sliceId) {
        std::string deprName = "VPU-" + std::to_string(sliceId);
        if (deviceName == deprName) {
            return sliceId;
        }
    }
    // *********************************************************************************

    IE_THROW() << "Device name conversion error - bad name: " << deviceName;
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
    const auto platformPos = deviceName.rfind('.');
    const auto platformName = (platformPos == std::string::npos) ? deviceName : deviceName.substr(0, platformPos);
    if (!isPlatformNameSupported(platformName)) {
        IE_THROW() << "Unexpected device name: " << deviceName;
    }

    return platformName;
}

ie::VPUXConfigParams::VPUXPlatform utils::getPlatformByDeviceName(const std::string& deviceName) {
    const auto platformName = getPlatformNameByDeviceName(deviceName);
    if (!isPlatformNameSupported(platformName)) {
        IE_THROW() << "Unexpected device name: " << deviceName;
    }

    return platformNameInverseMap.at(platformName);
}

bool utils::isPlatformNameSupported(const std::string& platformName) {
    return (platformNameInverseMap.find(platformName) != platformNameInverseMap.end());
}
