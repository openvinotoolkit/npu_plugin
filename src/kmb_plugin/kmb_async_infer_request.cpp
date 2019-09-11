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

#include <memory>
#include "kmb_async_infer_request.h"

using namespace vpu::KmbPlugin;
using namespace InferenceEngine;

KmbAsyncInferRequest::KmbAsyncInferRequest(const KmbInferRequest::Ptr &request,
                                                 const InferenceEngine::ITaskExecutor::Ptr &taskExecutorStart,
                                                 const InferenceEngine::ITaskExecutor::Ptr &taskExecutorGetResult,
                                                 const InferenceEngine::TaskSynchronizer::Ptr &taskSynchronizer,
                                                 const InferenceEngine::ITaskExecutor::Ptr &callbackExecutor,
                                                 const Logger::Ptr &log)
        : InferenceEngine::AsyncInferRequestThreadSafeDefault(request,
                                                              taskExecutorStart,
                                                              taskSynchronizer,
                                                              callbackExecutor),
          _logger(log), _request(request), _taskExecutorGetResult(taskExecutorGetResult) {}


InferenceEngine::StagedTask::Ptr KmbAsyncInferRequest::createAsyncRequestTask() {
    return std::make_shared<StagedTask>([this]() {
        auto asyncTaskCopy = _asyncTask;
        try {
            switch (asyncTaskCopy->getStage()) {
                case 3: {
                    _request->InferAsync();
                    asyncTaskCopy->stageDone();
                    _taskExecutorGetResult->startTask(asyncTaskCopy);
                }
                    break;
                case 2: {
                    _request->GetResult();
                    asyncTaskCopy->stageDone();
                    if (_callbackManager.isCallbackEnabled()) {
                        _callbackManager.startTask(asyncTaskCopy);
                    } else {
                        asyncTaskCopy->stageDone();
                    }
                }
                    break;
                case 1: {
                    setIsRequestBusy(false);
                    asyncTaskCopy->stageDone();
                    _callbackManager.runCallback();
                }
                    break;
                default:
                    break;
            }
        } catch (...) {
            processAsyncTaskFailure(asyncTaskCopy);
        }
    }, 3);
}

KmbAsyncInferRequest::~KmbAsyncInferRequest() {
    waitAllAsyncTasks();
}

