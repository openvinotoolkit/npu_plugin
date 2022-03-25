//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "HddlUnite.h"
#include "ie_extension.h"
#include <device_helpers.hpp>

namespace DeviceName {

std::string getName();
std::string getNameInPlugin();
std::set<std::string> getDevicesNames();
std::set<std::string> getDevicesNamesWithPrefix();

bool isEmulator();

inline std::set<std::string> getDevicesNames() {
    std::vector<HddlUnite::Device> devices;
    HddlStatusCode code = getAvailableDevices(devices);
    if (code != HDDL_OK || devices.empty()) {
        IE_THROW() << "No devices found";
    }
    std::set<std::string> deviceNames;
    for (const auto& device: devices) {
        deviceNames.insert(utils::getDeviceNameBySwDeviceId(device.getSwDeviceId()));
    }
    return deviceNames;
}

inline std::set<std::string> getDevicesNamesWithPrefix() {
    auto devices = getDevicesNames();
    std::set<std::string> namesWithPluginPrefix;
    for (const auto& device: devices) {
        namesWithPluginPrefix.insert("VPUX." + device);
    }
    return namesWithPluginPrefix;
}

inline std::string getName() {
    auto devices = getDevicesNames();
    if (devices.size() > 1) {
        IE_THROW() << "More than 1 device is not supported";
    }
    return *devices.begin();
}

inline bool isEmulator() {
    const std::string emulatorDeviceName = "127.0.0.1";
    const std::string deviceName = getName();
    if (deviceName.find(emulatorDeviceName) != std::string::npos) {
        return false;
    }
    return true;
}

inline std::string getNameInPlugin() {
    const std::string pluginName = "VPUX";
    return pluginName + "." + getName();
}
}