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

#pragma once

#include <cstdint>
#include <map>

#include "vpux_private_config.hpp"

namespace utils {
bool isVPUDevice(const uint32_t deviceId);
uint32_t getSliceIdBySwDeviceId(const uint32_t swDevId);
int getSliceIdByDeviceName(const std::string& deviceName);
InferenceEngine::VPUXConfigParams::VPUXPlatform getPlatformBySwDeviceId(const uint32_t swDevId);
std::string getPlatformNameByDeviceName(const std::string& deviceName);
bool isPlatformNameSupported(const std::string& platformName);
std::string getDeviceNameBySwDeviceId(const uint32_t swDevId);
InferenceEngine::VPUXConfigParams::VPUXPlatform getPlatformByDeviceName(const std::string& deviceName);
}  // namespace utils
