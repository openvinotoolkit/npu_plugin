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

#include <memory>
#include <string>
#include <vpux.hpp>

namespace vpux {

class EmulatorDevice final : public IDevice {
public:
    EmulatorDevice();
    std::shared_ptr<Executor> createExecutor(
        const NetworkDescription::Ptr& networkDescription, const VPUXConfig& config) override;

    virtual std::shared_ptr<Allocator> getAllocator() const {
        return nullptr;
    }
    std::string getName() const override;
private:
    std::unique_ptr<vpu::Logger> _logger;
};

}  // namespace vpux
