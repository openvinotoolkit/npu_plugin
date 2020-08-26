//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "hddl2_async_infer_request.h"

using namespace vpu::HDDL2Plugin;
namespace IE = InferenceEngine;

// clang-format off
HDDL2AsyncInferRequest::HDDL2AsyncInferRequest(const HDDL2InferRequest::Ptr &inferRequest,
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

HDDL2AsyncInferRequest::~HDDL2AsyncInferRequest() { StopAndWait(); }
