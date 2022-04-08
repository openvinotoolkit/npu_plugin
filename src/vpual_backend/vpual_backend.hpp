//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#pragma once

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"

#include <ie_common.h>

#include <map>
#include <memory>

namespace vpux {

class VpualEngineBackend final : public vpux::IEngineBackend {
    Logger _logger;
    std::map<std::string, std::shared_ptr<IDevice>> _devices;

public:
    VpualEngineBackend();
    const std::string getName() const override {
        return "VPUAL";
    }
    void registerOptions(OptionsDesc& options) const override;
    // TODO Investigate which device should be returned by getDevice without parameters
    const std::shared_ptr<IDevice> getDevice() const override;
    const std::shared_ptr<IDevice> getDevice(const std::string& deviceId) const override;
    const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& map) const override;
    const std::vector<std::string> getDeviceNames() const override;

private:
    const std::map<std::string, std::shared_ptr<IDevice>> createDeviceMap();
};

}  // namespace vpux
