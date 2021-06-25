//
// Copyright 2020 Intel Corporation.
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

#include "vpual_backend.hpp"

#include <ie_common.h>

#include <description_buffer.hpp>
#include <memory>

#if defined(__arm__) || defined(__aarch64__)
#include <mvMacros.h>
#include <xlink_uapi.h>
#endif

#include <device_helpers.hpp>

#include "vpual_device.hpp"
// [Track number: E#12122]
// TODO Remove this header after removing KMB deprecated parameters in future releases
#include "vpux/kmb_params.hpp"
#include "vpux/vpux_plugin_params.hpp"

namespace vpux {

namespace {

struct PlatformInfo {
    std::string _name;
    InferenceEngine::VPUXConfigParams::VPUXPlatform _platform;
};

#if defined(__arm__) || defined(__aarch64__)
std::shared_ptr<xlink_handle> getHandleById(const uint32_t& devId) {
    auto xlinkHandlePtr = std::make_shared<xlink_handle>();
    xlinkHandlePtr->sw_device_id = devId;
    xlinkHandlePtr->dev_type = VPUIP_DEVICE;
    return xlinkHandlePtr;
}

bool isDeviceFree(const std::shared_ptr<xlink_handle>& devHandle) {
    uint32_t devStatus = XLINK_DEV_ERROR;
    xlink_error getStatusResult = xlink_get_device_status(devHandle.get(), &devStatus);
    // FIXME this is a hack for detect + classify use case
    // for some reason two instances of IE Core is created (one for each network)
    // both networks run on the same device
    // the first instance of plug-in seizes the device, so the second instance receives device busy
    // [Track number: H#18012987025]
    return getStatusResult == X_LINK_SUCCESS;
}

PlatformInfo getPlatformInfo(const std::shared_ptr<xlink_handle>& devHandle) {
    PlatformInfo result;
    result._name = utils::getDeviceNameBySwDeviceId(devHandle->sw_device_id);
    result._platform = utils::getPlatformByDeviceName(result._name);
    return result;
}
#endif

std::vector<PlatformInfo> getAvailableDevices() {
    std::vector<PlatformInfo> platformInfoList;
#if defined(__arm__) || defined(__aarch64__)
    xlink_error initResult = xlink_initialize();
    if (initResult != X_LINK_SUCCESS) {
        IE_THROW() << "VpualExecutor::getDeviceList: xlink_inititalize failed with error: " << initResult;
    }

    // get all devices
    constexpr size_t maxDeviceListSize = 8;
    std::vector<uint32_t> deviceIdList(maxDeviceListSize, 0x0);
    uint32_t availableDevicesCount = 0;
    xlink_error getDevResult = xlink_get_device_list(deviceIdList.data(), &availableDevicesCount);
    if (getDevResult != X_LINK_SUCCESS) {
        IE_THROW() << "VpualExecutor::getDeviceList: xlink_get_device_list failed with error: " << getDevResult;
    }
    deviceIdList.resize(availableDevicesCount);

    // filter devices by type since VPUAL backend cannot use PCIe end-points for inference
    std::vector<uint32_t> vpuDevIdList;
    std::copy_if(deviceIdList.begin(), deviceIdList.end(), std::back_inserter(vpuDevIdList), utils::isVPUDevice);

    std::vector<std::shared_ptr<xlink_handle>> devHandleList;
    std::transform(vpuDevIdList.begin(), vpuDevIdList.end(), std::back_inserter(devHandleList), getHandleById);

    // filter devices by status
    std::vector<std::shared_ptr<xlink_handle>> freeDevIdList;
    std::copy_if(devHandleList.begin(), devHandleList.end(), std::back_inserter(freeDevIdList), isDeviceFree);

    // get names of free devices
    std::transform(freeDevIdList.begin(), freeDevIdList.end(), std::back_inserter(platformInfoList), getPlatformInfo);
#endif
    return platformInfoList;
}

}  // namespace

VpualEngineBackend::VpualEngineBackend()
    : _logger(std::unique_ptr<vpu::Logger>(
          // [Track number: S#42840]
          // TODO: config will come by another PR, for now let's use Error log level
          new vpu::Logger("VpualBackend", vpu::LogLevel::Error /*_config.logLevel()*/, vpu::consoleOutput()))),
      _devices(createDeviceMap()) {}

const std::map<std::string, std::shared_ptr<IDevice>> VpualEngineBackend::createDeviceMap() {
    auto deviceIds = getAvailableDevices();
    std::map<std::string, std::shared_ptr<IDevice>> devices;
    for (const auto& id : deviceIds) {
        devices.insert({id._name, std::make_shared<VpualDevice>(id._name, id._platform)});
        _logger->info("Device %s found.", id._name);
    }

    return devices;
}

const std::shared_ptr<IDevice> VpualEngineBackend::getDevice() const {
    if (_devices.empty()) {
        IE_THROW() << "There are no devices";
    }
    return _devices.begin()->second;
}

const std::shared_ptr<IDevice> VpualEngineBackend::getDevice(const std::string& deviceId) const {
    if (_devices.empty()) {
        IE_THROW() << "There are no devices";
    }

    try {
        // Platform and device are not provided - return first available device
        if (deviceId.empty()) {
            return _devices.begin()->second;
        }

        const auto expectedPlatformName = utils::getPlatformNameByDeviceName(deviceId);
        const auto currentPlatformName = utils::getPlatformNameByDeviceName(_devices.begin()->second->getName());
        if (expectedPlatformName != currentPlatformName) {
            _logger->warning("Device with platform %s not found", expectedPlatformName);
            return nullptr;
        }
        const auto expectedSliceId = utils::getSliceIdByDeviceName(deviceId);
        const std::string expectedDeviceName = expectedPlatformName + "." + std::to_string(expectedSliceId);
        return _devices.at(expectedDeviceName);
    } catch (...) {
        _logger->warning("Device %s not found", deviceId);
    }
    return nullptr;
}

const std::shared_ptr<IDevice> VpualEngineBackend::getDevice(const InferenceEngine::ParamMap& map) const {
    std::string deviceId;

    // [Track number: E#12122]
    // TODO Remove KMB_PARAM_KEY part after removing deprecated KMB parameters in future releases
    auto deprDeviceIdIter = map.find(InferenceEngine::KMB_PARAM_KEY(DEVICE_ID));
    if (deprDeviceIdIter != map.end()) {
        deviceId = deprDeviceIdIter->second.as<std::string>();
    } else {
        try {
            deviceId = map.at(InferenceEngine::VPUX_PARAM_KEY(DEVICE_ID)).as<std::string>();
        } catch (...) {
            IE_THROW() << "Device ID is not provided!";
        }
    }

    try {
        return getDevice(deviceId);
    } catch (...) {
        _logger->warning("Device %s not found", deviceId);
    }
    return nullptr;
}

const std::vector<std::string> VpualEngineBackend::getDeviceNames() const {
    std::vector<std::string> availableDevices;
    for (const auto& elem : _devices) {
        const auto& device = elem.second;
        availableDevices.emplace_back(device->getName());
    }
    return availableDevices;
}

}  // namespace vpux

INFERENCE_PLUGIN_API(void)
CreateVPUXEngineBackend(std::shared_ptr<vpux::IEngineBackend>& backend) {
    backend = std::make_shared<vpux::VpualEngineBackend>();
}
