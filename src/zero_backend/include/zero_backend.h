//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <map>
#include <memory>

#include "vpux.hpp"

namespace vpux {
class ZeroStructsInitializer;

class ZeroEngineBackend final : public vpux::IEngineBackend {
public:
    ZeroEngineBackend();
    virtual ~ZeroEngineBackend();
    virtual const std::shared_ptr<IDevice> getDevice() const override;
    virtual const std::shared_ptr<IDevice> getDevice(const std::string&) const override;
    const std::string getName() const override {
        return "LEVEL0";
    }
    const std::vector<std::string> getDeviceNames() const override;

private:
    std::unique_ptr<ZeroStructsInitializer> instance;
};

}  // namespace vpux
