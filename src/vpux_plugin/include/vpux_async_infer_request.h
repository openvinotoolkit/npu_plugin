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

#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "vpux_infer_request.h"

namespace vpux {

class AsyncInferRequest final : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<AsyncInferRequest>;

    explicit AsyncInferRequest(const InferRequest::Ptr& inferRequest,
                               const InferenceEngine::ITaskExecutor::Ptr& requestExecutor,
                               const InferenceEngine::ITaskExecutor::Ptr& getResultExecutor,
                               const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor);

    ~AsyncInferRequest();

private:
    InferRequest::Ptr _inferRequest;
    InferenceEngine::ITaskExecutor::Ptr _getResultExecutor;
};

}  // namespace vpux
