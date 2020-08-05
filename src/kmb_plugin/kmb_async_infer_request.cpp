//
// Copyright 2019 Intel Corporation.
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

#include "kmb_async_infer_request.h"

using namespace vpu::KmbPlugin;
using namespace InferenceEngine;

KmbAsyncInferRequest::KmbAsyncInferRequest(const KmbInferRequest::Ptr& request,
    const InferenceEngine::ITaskExecutor::Ptr& taskExecutorStart,
    const InferenceEngine::ITaskExecutor::Ptr& taskExecutorGetResult,
    const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor, const vpu::Logger::Ptr& log)
    : InferenceEngine::AsyncInferRequestThreadSafeDefault(request, taskExecutorStart, callbackExecutor),
      _logger(log),
      _request(request),
      _taskExecutorGetResult(taskExecutorGetResult) {
    _pipeline = {{_requestExecutor,
                     [this] {
                         _request->InferAsync();
                     }},
        {_taskExecutorGetResult, [this] {
             _request->GetResult();
         }}};
}

KmbAsyncInferRequest::~KmbAsyncInferRequest() { StopAndWait(); }
