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

// IE
#include <ie_metric_helpers.hpp>
// Plugin
#include "device_helpers.hpp"
#include "vpux/properties.hpp"
#include "vpux_metrics.h"
#include "vpux_private_config.hpp"
#include "vpux_private_properties.hpp"

namespace vpux {
namespace IE = InferenceEngine;

Metrics::Metrics(const VPUXBackends::CPtr& backends): _backends(backends) {
    _supportedMetrics = {
            METRIC_KEY(SUPPORTED_METRICS),         METRIC_KEY(AVAILABLE_DEVICES),
            METRIC_KEY(FULL_DEVICE_NAME),          METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(OPTIMIZATION_CAPABILITIES), METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS),
            METRIC_KEY(RANGE_FOR_STREAMS),         METRIC_KEY(IMPORT_EXPORT_SUPPORT),
            METRIC_KEY(DEVICE_ARCHITECTURE),
    };

    _supportedConfigKeys = {ov::log::level.name(),
                            ov::enable_profiling.name(),
                            ov::device::id.name(),
                            ov::hint::performance_mode.name(),
                            ov::num_streams.name(),
                            ov::intel_vpux::compilation_mode_params.name(),
                            ov::intel_vpux::inference_shaves.name()};
}

std::vector<std::string> Metrics::GetAvailableDevicesNames() const {
    return _backends == nullptr ? std::vector<std::string>() : _backends->getAvailableDevicesNames();
}

// TODO each backend may support different metrics
const std::vector<std::string>& Metrics::SupportedMetrics() const {
    return _supportedMetrics;
}

std::string Metrics::GetFullDeviceName(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    return utils::getFullDeviceNameByDeviceName(devName);
}

// TODO each backend may support different configs
const std::vector<std::string>& Metrics::GetSupportedConfigKeys() const {
    return _supportedConfigKeys;
}

// TODO each backend may support different optimization capabilities
const std::vector<std::string>& Metrics::GetOptimizationCapabilities() const {
    return _optimizationCapabilities;
}

const std::tuple<uint32_t, uint32_t, uint32_t>& Metrics::GetRangeForAsyncInferRequest() const {
    return _rangeForAsyncInferRequests;
}

const std::tuple<uint32_t, uint32_t>& Metrics::GetRangeForStreams() const {
    return _rangeForStreams;
}

std::string Metrics::GetDeviceArchitecture(const std::string& specifiedDeviceName) const {
    const auto devName = getDeviceName(specifiedDeviceName);
    return utils::getPlatformNameByDeviceName(devName);
}

std::string Metrics::GetBackendName() const {
    if (_backends == nullptr) {
        IE_THROW() << "No available backends";
    }

    return _backends->getBackendName();
}

std::string Metrics::getDeviceName(const std::string& specifiedDeviceName) const {
    std::vector<std::string> devNames;
    if (_backends == nullptr || (devNames = _backends->getAvailableDevicesNames()).empty()) {
        IE_THROW() << "No available devices";
    }

    // In case of single device and empty input from user we should use the first element from the device list
    if (specifiedDeviceName.empty()) {
        if (devNames.size() == 1) {
            return devNames[0];
        } else {
            IE_THROW() << "The device name was not specified. Please specify device name by providing DEVICE_ID";
        }
    }

    // In case of multiple devices and non-empty input from user we have to check if such device exists in system
    // First, check format "platform.slice_id"
    if (std::find(devNames.cbegin(), devNames.cend(), specifiedDeviceName) == devNames.cend()) {
        // Second, check format "platform"
        const auto userPlatform = utils::getPlatformByDeviceName(specifiedDeviceName);
        const auto platformIt = std::find_if(devNames.cbegin(), devNames.cend(), [=](const std::string& devName) {
            const auto devPlatform = utils::getPlatformByDeviceName(devName);
            return userPlatform == devPlatform;
        });
        if (platformIt == devNames.cend()) {
            IE_THROW() << "List of system devices doesn't contain specified device " << specifiedDeviceName;
        }
    }

    return specifiedDeviceName;
}

}  // namespace vpux
