//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#pragma once

#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <ie_input_info.hpp>

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"

namespace vpux {

class ZeroInferRequest : public IInferRequest {
public:
    using Ptr = std::shared_ptr<InferRequest>;

    explicit ZeroInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                              const InferenceEngine::OutputsDataMap& networkOutputs, const Executor::Ptr& executor,
                              const Config& config, const std::string& netName,
                              const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                              const std::vector<std::shared_ptr<const ov::Node>>& results,
                              const std::shared_ptr<InferenceEngine::IAllocator>& allocator = nullptr);

    void InferImpl() override;
    void InferAsync() override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    void GetResult() override;

protected:
    const Executor::Ptr _executorPtr;
    const Config _config;
    Logger _logger;
    std::shared_ptr<InferenceEngine::IAllocator> _allocator;
};

}  //  namespace vpux
