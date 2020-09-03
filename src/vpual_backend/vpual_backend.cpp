#include "vpual_backend.hpp"

#include <ie_common.h>

#include <description_buffer.hpp>
#include <memory>

#if defined(__arm__) || defined(__aarch64__)
#include <mvMacros.h>
#include <xlink_uapi.h>
#endif

#include "vpual_device.hpp"

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
    return (getStatusResult == X_LINK_SUCCESS && devStatus == XLINK_DEV_OFF);
}

std::string getNameByHandle(const std::shared_ptr<xlink_handle>& devHandle) {
    constexpr size_t maxDeviceNameSize = 128;
    std::vector<char> devNameData(maxDeviceNameSize, 0x0);
    xlink_error getNameResult = xlink_get_device_name(devHandle.get(), devNameData.data(), devNameData.size());
    if (getNameResult != X_LINK_SUCCESS) {
        THROW_IE_EXCEPTION << "getNameByDeviceId: xlink_get_device_name failed with error: " << getNameResult;
    }
    std::string devName = devNameData.data();
    static const std::map<std::string, std::string> xlinkNameMapping = {
        {"vpu-slice-0", "VPU-0"},
        {"vpu-slice-1", "VPU-1"},
        {"vpu-slice-2", "VPU-2"},
        {"vpu-slice-3", "VPU-3"},
    };
    return xlinkNameMapping.at(devName);
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

    std::vector<std::shared_ptr<xlink_handle>> devHandleList;
    std::transform(deviceIdList.begin(), deviceIdList.end(), std::back_inserter(devHandleList), getHandleById);

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
