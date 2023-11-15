//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux_async_infer_request.h"

namespace vpux {
namespace ie = InferenceEngine;

// clang-format off
AsyncInferRequest::AsyncInferRequest(const IInferRequest::Ptr &inferRequest,
                                     const ie::ITaskExecutor::Ptr &requestExecutor,
                                     const ie::ITaskExecutor::Ptr &getResultExecutor,
                                     const ie::ITaskExecutor::Ptr &callbackExecutor)
        : ie::AsyncInferRequestThreadSafeDefault(inferRequest, requestExecutor, callbackExecutor),
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
