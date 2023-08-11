//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"

#include "emulator_infer_request.hpp"

#include <memory>
#include <string>

namespace vpux {

class EmulatorDevice final : public IDevice {
public:
    EmulatorDevice();
    std::shared_ptr<Executor> createExecutor(const NetworkDescription::Ptr& networkDescription,
                                             const Config& config) override;

    std::shared_ptr<Allocator> getAllocator() const override {
        return nullptr;
    }
    std::shared_ptr<Allocator> getAllocator(const InferenceEngine::ParamMap& params) const override {
        return nullptr;
    }

    std::string getName() const override;

    IInferRequest::Ptr createInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                                          const InferenceEngine::OutputsDataMap& networkOutputs,
                                          const Executor::Ptr& executor, const Config& config,
                                          const std::string& networkName,
                                          const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                          const std::vector<std::shared_ptr<const ov::Node>>& results,
                                          const vpux::DataMap& networkStatesInfo,
                                          const std::shared_ptr<InferenceEngine::IAllocator>& allocator) override {
        return std::make_shared<EmulatorInferRequest>(networkInputs, networkOutputs, executor, config, networkName,
                                                      parameters, results, networkStatesInfo, allocator);
    }

private:
    Logger _logger;
};

}  // namespace vpux
