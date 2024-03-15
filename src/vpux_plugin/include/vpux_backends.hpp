//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

// System
#include <memory>
#include <set>
#include <vector>

// Plugin
#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux_private_properties.hpp"

namespace vpux {

/** @brief Represent container for all backends and hide all related searching logic */
class VPUXBackends final {
public:
    using Ptr = std::shared_ptr<VPUXBackends>;
    using CPtr = std::shared_ptr<const VPUXBackends>;

    explicit VPUXBackends(const std::vector<std::string>& backendRegistry, const Config& config);

    std::shared_ptr<Device> getDevice(const std::string& specificName = "") const;
    std::shared_ptr<Device> getDevice(const ov::AnyMap& paramMap) const;
    std::vector<std::string> getAvailableDevicesNames() const;
    std::string getBackendName() const;
    void registerOptions(OptionsDesc& options) const;
    std::string getCompilationPlatform(const InferenceEngine::VPUXConfigParams::VPUXPlatform platform,
                                       const std::string& deviceId) const;

    void setup(const Config& config);

private:
    Logger _logger;
    std::shared_ptr<EngineBackend> _backend;
};

}  // namespace vpux
