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

#include "kmb_metrics.h"

#if defined(__arm__) || defined(__aarch64__)
#include <mvMacros.h>
#include <xlink_uapi.h>
#endif

#include <algorithm>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/utils/error.hpp>

#include "kmb_private_config.hpp"

using namespace vpu::KmbPlugin;
using namespace InferenceEngine::VPUConfigParams;
using namespace InferenceEngine::PluginConfigParams;

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
//------------------------------------------------------------------------------
// Implementation of methods of class KmbMetrics
//------------------------------------------------------------------------------

KmbMetrics::KmbMetrics() {
    _supportedMetrics = {
        METRIC_KEY(SUPPORTED_METRICS),
        METRIC_KEY(AVAILABLE_DEVICES),
        METRIC_KEY(FULL_DEVICE_NAME),
        METRIC_KEY(SUPPORTED_CONFIG_KEYS),
        METRIC_KEY(OPTIMIZATION_CAPABILITIES),
        METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS),
        METRIC_KEY(RANGE_FOR_STREAMS),
    };

    _supportedConfigKeys = {
        VPU_KMB_CONFIG_KEY(PLATFORM),
        CONFIG_KEY(DEVICE_ID),
        CONFIG_KEY(LOG_LEVEL),
        VPU_KMB_CONFIG_KEY(LOAD_NETWORK_AFTER_COMPILATION),
        VPU_KMB_CONFIG_KEY(KMB_EXECUTOR),
        KMB_CONFIG_KEY(THROUGHPUT_STREAMS),
        VPU_KMB_CONFIG_KEY(PREPROCESSING_SHAVES),
        VPU_KMB_CONFIG_KEY(PREPROCESSING_LPI),
#ifdef ENABLE_M2I
        VPU_KMB_CONFIG_KEY(USE_M2I),
#endif
    };
}

std::vector<std::string> KmbMetrics::AvailableDevicesNames() const {
    std::vector<std::string> availableDevices = getAvailableDevices();

    std::sort(availableDevices.begin(), availableDevices.end());
    return availableDevices;
}

const std::vector<std::string>& KmbMetrics::SupportedMetrics() const { return _supportedMetrics; }

std::string KmbMetrics::GetFullDevicesNames() { return {"Gen3 Intel(R) Movidius(TM) VPU code-named Keem Bay"}; }

const std::vector<std::string>& KmbMetrics::GetSupportedConfigKeys() const { return _supportedConfigKeys; }

const std::vector<std::string>& KmbMetrics::GetOptimizationCapabilities() const { return _optimizationCapabilities; }

const std::tuple<uint32_t, uint32_t, uint32_t>& KmbMetrics::GetRangeForAsyncInferRequest() const {
    return _rangeForAsyncInferRequests;
}

const std::tuple<uint32_t, uint32_t>& KmbMetrics::GetRangeForStreams() const { return _rangeForStreams; }
