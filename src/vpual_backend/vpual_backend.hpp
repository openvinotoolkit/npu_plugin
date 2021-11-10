//
// Copyright 2020 Intel Corporation.
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

#include "vpux.hpp"

#include <ie_common.h>
#include <vpu/utils/logger.hpp>

#include <map>
#include <memory>

namespace vpux {

class VpualEngineBackend final : public vpux::IEngineBackend {
    std::unique_ptr<vpu::Logger> _logger;
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
