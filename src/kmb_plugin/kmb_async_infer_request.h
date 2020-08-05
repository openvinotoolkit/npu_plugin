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

#pragma once

#include "cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp"
#include "kmb_infer_request.h"

namespace vpu {
namespace KmbPlugin {

class KmbAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    KmbAsyncInferRequest(const KmbInferRequest::Ptr& request,
        const InferenceEngine::ITaskExecutor::Ptr& taskExecutorStart,
        const InferenceEngine::ITaskExecutor::Ptr& taskExecutorGetResult,
        const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor, const vpu::Logger::Ptr& log);

    ~KmbAsyncInferRequest();

private:
    Logger::Ptr _logger;
    KmbInferRequest::Ptr _request;
    InferenceEngine::ITaskExecutor::Ptr _taskExecutorGetResult;
};

}  // namespace KmbPlugin
}  // namespace vpu
