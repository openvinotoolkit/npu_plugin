//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cstdint>
#include <map>

#include "vpux_private_properties.hpp"

namespace utils {
bool isVPUDevice(const uint32_t deviceId);
uint32_t getSliceIdBySwDeviceId(const uint32_t swDevId);
int getSliceIdByDeviceName(const std::string& deviceName);
InferenceEngine::VPUXConfigParams::VPUXPlatform getPlatformBySwDeviceId(const uint32_t swDevId);
std::string getPlatformNameByDeviceName(const std::string& deviceName);
std::string getFullDeviceNameByDeviceName(const std::string& deviceName);
bool isPlatformNameSupported(const std::string& platformName);
std::string getDeviceNameBySwDeviceId(const uint32_t swDevId);
InferenceEngine::VPUXConfigParams::VPUXPlatform getPlatformByDeviceName(const std::string& deviceName);
// TODO Remove after removing deprecated device names from VPUAL backend
bool isDeviceNameVpualDeprecated(const std::string& deviceName);
}  // namespace utils
