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

#include "vpux_async_infer_request.h"

namespace vpux {
namespace IE = InferenceEngine;

// clang-format off
AsyncInferRequest::AsyncInferRequest(const InferRequest::Ptr &inferRequest,
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

AsyncInferRequest::~AsyncInferRequest() { StopAndWait(); }

} // namespace vpux
