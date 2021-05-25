//
// Copyright 2019 Intel Corporation.
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

#include "HddlUnite.h"
#include "ie_extension.h"

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
        deviceNames.insert(device.getName());
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