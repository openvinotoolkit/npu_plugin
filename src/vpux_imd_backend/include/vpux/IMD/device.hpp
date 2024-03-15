//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "infer_request.hpp"
#include "vpux_private_properties.hpp"

namespace vpux {

class IMDDevice final : public IDevice {
public:
    explicit IMDDevice(InferenceEngine::VPUXConfigParams::VPUXPlatform platform);

public:
    std::shared_ptr<Executor> createExecutor(const NetworkDescription::CPtr network, const Config& config) override;

    std::string getName() const override;
    std::string getFullDeviceName() const override;

    std::shared_ptr<SyncInferRequest> createInferRequest(
            const std::shared_ptr<const ov::ICompiledModel> compiledModel,
            const std::shared_ptr<const NetworkDescription> networkDescription, const Executor::Ptr executor,
            const Config& config) override {
        return std::make_shared<IMDInferRequest>(compiledModel, networkDescription, executor, config);
    }

private:
    InferenceEngine::VPUXConfigParams::VPUXPlatform _platform;
};

}  // namespace vpux
