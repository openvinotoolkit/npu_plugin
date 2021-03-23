//
// Copyright 2019 Intel Corporation.
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