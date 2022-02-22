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

const static std::map<uint32_t, InferenceEngine::VPUXConfigParams::VPUXPlatform> platformIdMap = {
        {0, InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700},  // KMB A0 / B0
        {1, InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3800},  // TBH prime
        {2, InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3900},  // TBH full
        {3, InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720},  // MTL
};

const static std::map<InferenceEngine::VPUXConfigParams::VPUXPlatform, std::string> platformNameMap = {
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO, "AUTO"},           // auto detection
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400_A0, "3400_A0"},  // KMB A0
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400, "3400"},        // KMB B0 500 MHz
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700, "3700"},        // KMB B0 700 MHz
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3800, "3800"},        // TBH Prime
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3900, "3900"},        // TBH Full
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720, "3720"},        // MTL
};

const static std::map<std::string, InferenceEngine::VPUXConfigParams::VPUXPlatform> platformNameInverseMap = {
        {"AUTO", InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO},             // auto detection
        {"3400_A0", InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400_A0},    // KMB A0
        {"3400", InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400},          // KMB B0 500 MHz
        {"3700", InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700},          // KMB B0 700 MHz
        {"3800", InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3800},          // TBH Prime
        {"3900", InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3900},          // TBH Full
        {"3720", InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720},          // MTL
        {"3400_A0_EMU", InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR},  // Emulator KMB A0
        {"3400_EMU", InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR},     // Emulator KMB B0 500 MHz
        {"3700_EMU", InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR},     // Emulator KMB B0 700 MHz
        {"3800_EMU", InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR},     // Emulator TBH Prime
        {"3900_EMU", InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR},     // Emulator TBH Full
        {"3720_EMU", InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR},     // Emulator MTL
};

// TODO Need to clarify the full names of devices. Definitely for MTL, possibly for others
const static std::map<InferenceEngine::VPUXConfigParams::VPUXPlatform, std::string> platformToFullDeviceNameMap = {
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400_A0,
         "Gen3 Intel(R) Movidius(TM) VPU 3400VE"},  // KMB A0
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3400,
         "Gen3 Intel(R) Movidius(TM) VPU 3400VE"},  // KMB B0 500 MHz
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700,
         "Gen3 Intel(R) Movidius(TM) VPU 3700VE"},  // KMB B0 700 MHz
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3800,
         "Gen3 Intel(R) Movidius(TM) S VPU 3800V"},  // TBH Prime
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3900,
         "Gen3 Intel(R) Movidius(TM) S VPU 3900V"},  // TBH Full
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720, "Gen4 Intel(R) Movidius(TM) VPU 3720VE"},  // MTL
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::EMULATOR, "Emulator"},  // Emulator
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

    int sliceId = 0;
    const int minSliceId = 0;
    const int maxSliceId = 3;
    // TODO Remove this part after removing deprecated device names in future releases
    // Check deprecated "VPU-slice_id" naming format
    // *********************************************************************************
    for (sliceId = minSliceId; sliceId <= maxSliceId; ++sliceId) {
        std::string deprName = std::string("VPU-") + std::to_string(sliceId);
        if (deviceName == deprName) {
            return sliceId;
        }
    }
    // *********************************************************************************

    // Check "only platform" naming format. For it return the first slice as well
    const auto platformName = utils::getPlatformNameByDeviceName(deviceName);
    if (platformName == deviceName) {
        return 0;
    }

    // Check "platform.slice_id" naming format
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

    IE_THROW() << "Device name conversion error - bad name: " << deviceName;
}

InferenceEngine::VPUXConfigParams::VPUXPlatform utils::getPlatformBySwDeviceId(const uint32_t swDevId) {
    // bits 7-4 define platform
    // right shift to omit bits 0-3. after that platform code is stored in bits 3-0
    // apply b1111 mask to discard anything but platform code
    uint32_t platformId = (swDevId >> 4) & 0xf;
    return platformIdMap.at(platformId);
}

std::string utils::getDeviceNameBySwDeviceId(const uint32_t swDevId) {
    InferenceEngine::VPUXConfigParams::VPUXPlatform platform = getPlatformBySwDeviceId(swDevId);
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

InferenceEngine::VPUXConfigParams::VPUXPlatform utils::getPlatformByDeviceName(const std::string& deviceName) {
    const auto platformName = getPlatformNameByDeviceName(deviceName);
    if (!isPlatformNameSupported(platformName)) {
        IE_THROW() << "Unexpected device name: " << deviceName;
    }

    return platformNameInverseMap.at(platformName);
}

std::string utils::getFullDeviceNameByDeviceName(const std::string& deviceName) {
    const auto platform = getPlatformByDeviceName(deviceName);
    if (platformToFullDeviceNameMap.find(platform) == platformToFullDeviceNameMap.end()) {
        IE_THROW() << "Unexpected device name: " << deviceName;
    }
    return platformToFullDeviceNameMap.at(platform);
}

bool utils::isPlatformNameSupported(const std::string& platformName) {
    return (platformNameInverseMap.find(platformName) != platformNameInverseMap.end());
}

// TODO Remove after removing deprecated device names from VPUAL backend
bool utils::isDeviceNameVpualDeprecated(const std::string& deviceName) {
    const int minSliceId = 0;
    const int maxSliceId = 3;

    for (auto sliceId = minSliceId; sliceId <= maxSliceId; ++sliceId) {
        std::string deprName = std::string("VPU-") + std::to_string(sliceId);
        if (deviceName == deprName) {
            return true;
        }
    }

    return false;
}
