//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {
namespace hddl2 {

class HDDL2Backend final : public vpux::IEngineBackend {
public:
    using Ptr = std::shared_ptr<HDDL2Backend>;
    using CPtr = std::shared_ptr<const HDDL2Backend>;

    explicit HDDL2Backend();

    const std::shared_ptr<IDevice> getDevice() const override;
    const std::shared_ptr<IDevice> getDevice(const std::string& specificDeviceName) const override;
    const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& map) const override;
    const std::vector<std::string> getDeviceNames() const override;
    const std::string getName() const override {
        return "HDDL2";
    }

    // TODO remove static and make them private
    static bool isServiceAvailable(Logger logger = Logger::global());
    static bool isServiceRunning();

private:
    Logger _logger;
    std::map<std::string, std::shared_ptr<IDevice>> _devices;
    std::map<std::string, std::shared_ptr<IDevice>> createDeviceMap();
};

}  // namespace hddl2
}  // namespace vpux
