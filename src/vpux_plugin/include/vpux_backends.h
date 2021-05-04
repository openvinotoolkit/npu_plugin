//
// Copyright 2020-2021 Intel Corporation.
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
    std::string getCompilationPlatform(const InferenceEngine::VPUXConfigParams::VPUXPlatform platform) const;

    void setup(const VPUXConfig& config);

private:
    vpu::Logger _logger;
    std::shared_ptr<EngineBackend> _backend;
};

}  // namespace vpux
