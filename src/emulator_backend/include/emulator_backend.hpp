//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux.hpp"

#include <memory>
#include <string>
#include <vector>

namespace vpux {

class EmulatorBackend final : public IEngineBackend {
public:
    EmulatorBackend();
    const std::string getName() const override {
        return "EMULATOR";
    }
    const std::shared_ptr<IDevice> getDevice() const override;
    const std::shared_ptr<IDevice> getDevice(const std::string& name) const override;
    const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& map) const override;
    const std::vector<std::string> getDeviceNames() const override;

private:
    std::shared_ptr<IDevice> _device;
};

}  // namespace vpux
