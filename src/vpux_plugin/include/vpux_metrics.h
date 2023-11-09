//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// System
#include <openvino/runtime/properties.hpp>
#include <string>
#include <unordered_set>
#include <vector>
#include <vpux.hpp>
// Plugin
#include "vpux_backends.h"
#include "vpux_private_properties.hpp"
// TODO should not be here
#include <vpux/vpux_plugin_config.hpp>

namespace vpux {

class Metrics final {
public:
    Metrics(const VPUXBackends::CPtr& backends);

    std::vector<std::string> GetAvailableDevicesNames() const;
    const std::vector<std::string>& SupportedMetrics() const;
    std::string GetFullDeviceName(const std::string& specifiedDeviceName) const;
    Uuid GetDeviceUuid(const std::string& specifiedDeviceName) const;
    const std::vector<std::string>& GetSupportedConfigKeys() const;
    const std::vector<std::string>& GetOptimizationCapabilities() const;
    const std::tuple<uint32_t, uint32_t, uint32_t>& GetRangeForAsyncInferRequest() const;
    const std::tuple<uint32_t, uint32_t>& GetRangeForStreams() const;
    std::string GetDeviceArchitecture(const std::string& specifiedDeviceName) const;
    std::string GetBackendName() const;
    uint64_t GetDeviceTotalMemSize(const std::string& specifiedDeviceName) const;
    uint32_t GetDriverVersion(const std::string& specifiedDeviceName) const;

    std::vector<ov::PropertyName> GetCachingProperties() const;

    ~Metrics() = default;

private:
    const VPUXBackends::CPtr _backends;
    std::vector<std::string> _supportedMetrics;
    std::vector<std::string> _supportedConfigKeys;
    const std::vector<std::string> _optimizationCapabilities = {
            METRIC_VALUE(FP16),                     //
            METRIC_VALUE(INT8),                     //
            ov::device::capability::EXPORT_IMPORT,  //
    };
    const std::vector<ov::PropertyName> _cachingProperties = {
            ov::device::architecture.name(),         ov::intel_vpux::compilation_mode_params.name(),
            ov::intel_vpux::dma_engines.name(),      ov::intel_vpux::dpu_groups.name(),
            ov::intel_vpux::compilation_mode.name(), ov::intel_vpux::driver_version.name(),
            ov::intel_vpux::compiler_type.name(),    ov::intel_vpux::use_elf_compiler_backend.name()};

    // Metric to provide a hint for a range for number of async infer requests. (bottom bound, upper bound, step)
    const std::tuple<uint32_t, uint32_t, uint32_t> _rangeForAsyncInferRequests{1u, 10u, 1u};

    // Metric to provide information about a range for streams.(bottom bound, upper bound)
    const std::tuple<uint32_t, uint32_t> _rangeForStreams{1u, 4u};

    std::string getDeviceName(const std::string& specifiedDeviceName) const;
};

}  // namespace vpux
