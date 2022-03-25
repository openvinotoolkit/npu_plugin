//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux.hpp"

namespace vpux {
namespace IMD {

class BackendImpl final : public IEngineBackend {
public:
    const std::shared_ptr<IDevice> getDevice() const override;
    const std::shared_ptr<IDevice> getDevice(const std::string& name) const override;
    const std::shared_ptr<IDevice> getDevice(const InferenceEngine::ParamMap& params) const override;

    const std::vector<std::string> getDeviceNames() const override;

    const std::string getName() const override;

    void registerOptions(OptionsDesc& options) const override;
};

}  // namespace IMD
}  // namespace vpux
