//
// Copyright 2020-2021 Intel Corporation.
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
// System
#include <memory>
#include <set>
#include <vector>
// Plugin
#include "vpux.hpp"

namespace vpux {

/** @brief Represent container for all backends and hide all related searching logic */
class VPUXBackends final {
public:
    using Ptr = std::shared_ptr<VPUXBackends>;
    using CPtr = std::shared_ptr<const VPUXBackends>;

    explicit VPUXBackends(const std::vector<std::string>& backendRegistry);

    std::shared_ptr<Device> getDevice(const std::string& specificName = "") const;
    std::shared_ptr<Device> getDevice(const InferenceEngine::ParamMap& paramMap) const;
    std::shared_ptr<Device> getDevice(const InferenceEngine::RemoteContext::Ptr& context) const;
    std::vector<std::string> getAvailableDevicesNames() const;
    std::string getBackendName() const;
    std::unordered_set<std::string> getSupportedOptions() const;
    std::string getCompilationPlatform(const InferenceEngine::VPUXConfigParams::VPUXPlatform platform,
                                       const std::string& deviceId) const;

    void setup(const VPUXConfig& config);

private:
    vpu::Logger _logger;
    std::shared_ptr<EngineBackend> _backend;
};

}  // namespace vpux
