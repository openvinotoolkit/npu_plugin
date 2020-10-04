//
// Copyright 2020 Intel Corporation.
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

// Plugin
#include "hddl2_exceptions.h"
#include "hddl2_metrics.h"
// Subplugin
#include <memory>

#include "subplugin/hddl2_backend.h"
#include "subplugin/hddl2_device.h"

using namespace vpux::HDDL2;
namespace vpux {
namespace HDDL2 {

HDDL2Backend::HDDL2Backend(const VPUXConfig& config)
    : _logger(std::make_shared<vpu::Logger>("HDDL2Backend", config.logLevel(), vpu::consoleOutput())),
      _devices(createDeviceMap()) {}

std::map<std::string, std::shared_ptr<vpux::IDevice>> HDDL2Backend::createDeviceMap() {
    std::map<std::string, std::shared_ptr<IDevice>> devices;
    // TODO Add more logs and cases handling
    if (vpu::HDDL2Plugin::HDDL2Metrics::isServiceAvailable(_logger) &&
        !vpu::HDDL2Plugin::HDDL2Metrics::GetAvailableDevicesNames().empty()) {
        devices.insert({"HDDL2", std::make_shared<HDDLUniteDevice>()});
        _logger->debug("HDDL2 device found for execution.");
    } else {
        _logger->debug("HDDL2 device not found for execution.");
    }
    return devices;
}

const std::shared_ptr<vpux::IDevice> HDDL2Backend::getDevice(const std::string& deviceName) const {
    const auto devices = vpu::HDDL2Plugin::HDDL2Metrics::GetAvailableDevicesNames();
    const auto it = std::find(devices.cbegin(), devices.cend(), deviceName);
    if (it != devices.end()) {
        return std::make_shared<HDDLUniteDevice>(*it);
    } else {
        return nullptr;
    }
}

}  // namespace HDDL2
}  // namespace vpux
