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

#include "vpual_backend.hpp"

#include <ie_common.h>

#include <description_buffer.hpp>
#include <memory>

#if defined(__arm__) || defined(__aarch64__)
#include <mvMacros.h>
#include <xlink_uapi.h>
#endif

#include "vpual_device.hpp"
#include "vpux/kmb_params.hpp"

namespace vpux {

namespace {

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

std::string getNameByHandle(const std::shared_ptr<xlink_handle>& devHandle) {
    // bits 3-1 define slice ID
    // right shift to omit bit 0, thus slice id is stored in bits 2-0
    // apply b111 mask to discard anything but slice ID
    uint32_t sliceId = (devHandle->sw_device_id >> 1) & 0x7;
    return "VPU-" + std::to_string(sliceId);
}

bool isVPUDevice(const uint32_t deviceId) {
    // bits 26-24 define interface type
    // 000 - IPC
    // 001 - PCIe
    // 010 - USB
    // 011 - ethernet
    constexpr uint32_t INTERFACE_TYPE_SELECTOR = 0x7000000;
    uint32_t interfaceType = (deviceId & INTERFACE_TYPE_SELECTOR);
    return (interfaceType == 0);
}
#endif

std::vector<std::string> getAvailableDevices() {
    std::vector<std::string> deviceNameList;
#if defined(__arm__) || defined(__aarch64__)
    xlink_error initResult = xlink_initialize();
    if (initResult != X_LINK_SUCCESS) {
        THROW_IE_EXCEPTION << "VpualExecutor::getDeviceList: xlink_inititalize failed with error: " << initResult;
    }

    // get all devices
    constexpr size_t maxDeviceListSize = 8;
    std::vector<uint32_t> deviceIdList(maxDeviceListSize, 0x0);
    uint32_t availableDevicesCount = 0;
    xlink_error getDevResult = xlink_get_device_list(deviceIdList.data(), &availableDevicesCount);
    if (getDevResult != X_LINK_SUCCESS) {
        THROW_IE_EXCEPTION << "VpualExecutor::getDeviceList: xlink_get_device_list failed with error: " << getDevResult;
    }
    deviceIdList.resize(availableDevicesCount);

    // filter devices by type since VPUAL backend cannot use PCIe end-points for inference
    std::vector<uint32_t> vpuDevIdList;
    std::copy_if(deviceIdList.begin(), deviceIdList.end(), std::back_inserter(vpuDevIdList), isVPUDevice);

    std::vector<std::shared_ptr<xlink_handle>> devHandleList;
    std::transform(vpuDevIdList.begin(), vpuDevIdList.end(), std::back_inserter(devHandleList), getHandleById);

    // filter devices by status
    std::vector<std::shared_ptr<xlink_handle>> freeDevIdList;
    std::copy_if(devHandleList.begin(), devHandleList.end(), std::back_inserter(freeDevIdList), isDeviceFree);

    // get names of free devices
    std::transform(freeDevIdList.begin(), freeDevIdList.end(), std::back_inserter(deviceNameList), getNameByHandle);
#endif
    return deviceNameList;
}

}  // namespace

VpualEngineBackend::VpualEngineBackend()
    : _logger(std::unique_ptr<vpu::Logger>(
          // TODO: config will come by another PR, for now let's use Error log level
          new vpu::Logger("VpualBackend", vpu::LogLevel::Error /*_config.logLevel()*/, vpu::consoleOutput()))),
      _devices(createDeviceMap()) {}

const std::map<std::string, std::shared_ptr<IDevice>> VpualEngineBackend::createDeviceMap() {
    auto deviceIds = getAvailableDevices();
    std::map<std::string, std::shared_ptr<IDevice>> devices;
    for (const auto& id : deviceIds) {
        devices.insert({id, std::make_shared<VpualDevice>(id)});
        _logger->info("Device %s found.", id);
    }

    return devices;
}

const std::map<std::string, std::shared_ptr<IDevice>>& VpualEngineBackend::getDevices() const { return _devices; }

const std::shared_ptr<IDevice> VpualEngineBackend::getDevice() const {
    if (_devices.empty()) {
        THROW_IE_EXCEPTION << "There are no any devices!";
    }
    return _devices.begin()->second;
}

const std::shared_ptr<IDevice> VpualEngineBackend::getDevice(const std::string& deviceId) const {
    try {
        return getDevices().at(deviceId);
    } catch (...) {
        _logger->warning("Device %s not found", deviceId);
    }
    return nullptr;
}

const std::shared_ptr<IDevice> VpualEngineBackend::getDevice(const InferenceEngine::ParamMap& map) const {
    std::string deviceId;
    try {
        deviceId = map.at(InferenceEngine::KMB_PARAM_KEY(DEVICE_ID)).as<std::string>();
    } catch (...) {
        THROW_IE_EXCEPTION << "Device ID is not provided!";
    }
    try {
        return getDevices().at(deviceId);
    } catch (...) {
        _logger->warning("Device %s not found", deviceId);
    }
    return nullptr;
}

const std::vector<std::string> VpualEngineBackend::getDeviceNames() const {
    std::vector<std::string> availableDevices;
    const auto& devices = getDevices();
    for (const auto& elem : devices) {
        const auto& device = elem.second;
        availableDevices.emplace_back(device->getName());
    }
    return availableDevices;
}

}  // namespace vpux

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode)
CreateVPUXEngineBackend(vpux::IEngineBackend*& backend, InferenceEngine::ResponseDesc* resp) noexcept {
    try {
        backend = new vpux::VpualEngineBackend();
        return InferenceEngine::StatusCode::OK;
    } catch (std::exception& ex) {
        return InferenceEngine::DescriptionBuffer(InferenceEngine::StatusCode::GENERAL_ERROR, resp) << ex.what();
    }
}
