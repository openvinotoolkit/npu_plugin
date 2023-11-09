//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"

#include <emu/manager.hpp>

namespace vpux {
class EmulatorInferRequest final : public IInferRequest {
public:
    using Ptr = std::shared_ptr<EmulatorInferRequest>;

    explicit EmulatorInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                                  const InferenceEngine::OutputsDataMap& networkOutputs, const Executor::Ptr& executor,
                                  const Config& config, const std::string& netName,
                                  const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                                  const std::vector<std::shared_ptr<const ov::Node>>& results,
                                  const vpux::NetworkIOVector& networkStatesInfo,
                                  const std::shared_ptr<InferenceEngine::IAllocator>& allocator = nullptr);

    void InferImpl() override;
    void InferAsync() override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    void GetResult() override;

private:
    void push(const InferenceEngine::BlobMap& inputs);
    void pull(InferenceEngine::BlobMap& outputs);

    const Executor::Ptr _executorPtr;
    const Config _config;
    Logger _logger;
    std::shared_ptr<InferenceEngine::IAllocator> _allocator;

    mv::emu::Manager _manager;
};

}  //  namespace vpux
