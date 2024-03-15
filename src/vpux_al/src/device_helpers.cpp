//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <openvino/core/except.hpp>

#include <device_helpers.hpp>

const static std::map<uint32_t, InferenceEngine::VPUXConfigParams::VPUXPlatform> platformIdMap = {
        {0, InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700},  // VPU30XX
        {1, InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720},  // VPU37XX
};

const static std::map<InferenceEngine::VPUXConfigParams::VPUXPlatform, std::string> platformNameMap = {
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO_DETECT, "AUTO_DETECT"},  // Auto detection
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700, "3700"},             // VPU30XX
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720, "3720"},             // VPU37XX
};

const static std::map<std::string, InferenceEngine::VPUXConfigParams::VPUXPlatform> platformNameInverseMap = {
        {"AUTO_DETECT", InferenceEngine::VPUXConfigParams::VPUXPlatform::AUTO_DETECT},  // Auto detection
        {"3700", InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700},             // VPU30XX
        {"3720", InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720},             // VPU37XX
};

const static std::map<InferenceEngine::VPUXConfigParams::VPUXPlatform, std::string> platformToFullDeviceNameMap = {
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3700, "Intel(R) NPU (3700VE)"},
        {InferenceEngine::VPUXConfigParams::VPUXPlatform::VPU3720, "Intel(R) NPU (3720VE)"},
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
            OPENVINO_THROW("Device name conversion error - ", ex.what());
        }
        if ((sliceId < minSliceId) || (sliceId > maxSliceId)) {
            OPENVINO_THROW("Device name conversion error - bad slice number: ", sliceId);
        }
        return sliceId;
    }

    OPENVINO_THROW("Device name conversion error - bad name: ", deviceName);
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
        OPENVINO_THROW("Unexpected device name: ", deviceName);
    }

    return platformName;
}

InferenceEngine::VPUXConfigParams::VPUXPlatform utils::getPlatformByDeviceName(const std::string& deviceName) {
    const auto platformName = getPlatformNameByDeviceName(deviceName);
    if (!isPlatformNameSupported(platformName)) {
        OPENVINO_THROW("Unexpected device name: ", deviceName);
    }

    return platformNameInverseMap.at(platformName);
}

std::string utils::getFullDeviceNameByDeviceName(const std::string& deviceName) {
    const auto platform = getPlatformByDeviceName(deviceName);
    if (platformToFullDeviceNameMap.find(platform) == platformToFullDeviceNameMap.end()) {
        OPENVINO_THROW("Unexpected device name: ", deviceName);
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
