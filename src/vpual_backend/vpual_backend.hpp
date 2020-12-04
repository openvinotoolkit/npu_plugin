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
#pragma once

#include <ie_common.h>

#include <map>
#include <memory>
#include <vpux.hpp>

#include "vpual_config.hpp"

namespace vpux {

class VpualEngineBackend final : public vpux::IEngineBackend {
    std::unique_ptr<vpu::Logger> _logger;
    std::map<std::string, std::shared_ptr<IDevice>> _devices;

public:
    VpualEngineBackend();
    const std::map<std::string, std::shared_ptr<IDevice>>& getDevices() const override;
    const std::string getName() const override { return "VPUAL"; }
    std::unordered_set<std::string> getSupportedOptions() const override { return _config.getRunTimeOptions(); }
    // TODO Investigate which device should be returned by getDevice without parameters
    const std::shared_ptr<IDevice> getDevice() const override;
    const std::shared_ptr<IDevice> getDevice(const std::string& deviceId) const override;
    const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& map) const override;
    const std::vector<std::string> getDeviceNames() const override;

private:
    const std::map<std::string, std::shared_ptr<IDevice>> createDeviceMap();
    VpualConfig _config;
};

}  // namespace vpux
