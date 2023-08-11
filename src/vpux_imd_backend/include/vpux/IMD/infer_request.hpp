//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <cpp_interfaces/interface/ie_ivariable_state_internal.hpp>
#include <ie_input_info.hpp>
#include <mutex>

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"
#include "vpux/utils/core/small_string.hpp"
#include "vpux/utils/core/small_vector.hpp"
#include "vpux/utils/core/string_ref.hpp"
#include "vpux_private_config.hpp"

#include <string>

namespace vpux {
namespace IMD {

class IMDInferRequest final : public IInferRequest {
public:
    using Ptr = std::shared_ptr<IMDInferRequest>;

    explicit IMDInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                             const InferenceEngine::OutputsDataMap& networkOutputs, const Executor::Ptr& executor,
                             const Config& config, const std::string& netName,
                             const std::vector<std::shared_ptr<const ov::Node>>& parameters,
                             const std::vector<std::shared_ptr<const ov::Node>>& results,
                             const vpux::DataMap& networkStatesInfo,
                             const std::shared_ptr<InferenceEngine::IAllocator>& allocator = nullptr);

    void InferImpl() override;
    void InferAsync() override;
    std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> QueryState() override;

    void GetResult() override;

private:
    SmallString createTempWorkDir();
    void storeNetworkBlob(StringRef workDir);
    void storeNetworkInputs(StringRef workDir, const InferenceEngine::BlobMap& inputs);
    void runApp(StringRef workDir);
    void loadNetworkOutputs(StringRef workDir, const InferenceEngine::BlobMap& outputs);

    void pull(const InferenceEngine::BlobMap& inputs, InferenceEngine::BlobMap& outputs);

    const Executor::Ptr _executorPtr;
    const Config _config;
    Logger _logger;
    std::shared_ptr<InferenceEngine::IAllocator> _allocator;

    const vpux::DataMap _statesInfo;
    std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> _states{};
    std::once_flag _fillStatesOnceFlag;
};

}  // namespace IMD
}  //  namespace vpux
