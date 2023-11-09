//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "vpux.hpp"

namespace vpux {

class AsyncInferRequest final : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<AsyncInferRequest>;

    explicit AsyncInferRequest(const IInferRequest::Ptr& inferRequest,
                               const InferenceEngine::ITaskExecutor::Ptr& requestExecutor,
                               const InferenceEngine::ITaskExecutor::Ptr& getResultExecutor,
                               const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor);
    AsyncInferRequest(const AsyncInferRequest&) = delete;
    AsyncInferRequest& operator=(const AsyncInferRequest&) = delete;
    ~AsyncInferRequest();

private:
    IInferRequest::Ptr _inferRequest;
    InferenceEngine::ITaskExecutor::Ptr _getResultExecutor;
};

}  // namespace vpux
