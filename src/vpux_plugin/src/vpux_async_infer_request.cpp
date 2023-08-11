//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux_async_infer_request.h"

namespace vpux {
namespace IE = InferenceEngine;

// clang-format off
AsyncInferRequest::AsyncInferRequest(const IInferRequest::Ptr &inferRequest,
                                     const IE::ITaskExecutor::Ptr &requestExecutor,
                                     const IE::ITaskExecutor::Ptr &getResultExecutor,
                                     const IE::ITaskExecutor::Ptr &callbackExecutor)
        : IE::AsyncInferRequestThreadSafeDefault(inferRequest, requestExecutor, callbackExecutor),
          _inferRequest(inferRequest), _getResultExecutor(getResultExecutor) {
    _pipeline = {
            {_requestExecutor,       [this] { _inferRequest->InferAsync(); }},
            {_getResultExecutor,     [this] { _inferRequest->GetResult(); }}
    };
}
// clang-format on

AsyncInferRequest::~AsyncInferRequest() {
    StopAndWait();
}

}  // namespace vpux
