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

#include <map>
#include <memory>
#include <vpux.hpp>

#include "zero_config.h"

namespace vpux {

class ZeroEngineBackend final : public vpux::IEngineBackend {
public:
    ZeroEngineBackend() = default;
    virtual const std::shared_ptr<IDevice> getDevice() const override;
    const std::string getName() const override {
        return "dKMB";
    }
    const std::vector<std::string> getDeviceNames() const override;
    std::unordered_set<std::string> getSupportedOptions() const override {
        return _config.getRunTimeOptions();
    }

private:
    ZeroConfig _config;
};

}  // namespace vpux
