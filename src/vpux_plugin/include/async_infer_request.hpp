//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/runtime/iasync_infer_request.hpp>
#include "sync_infer_request.hpp"

namespace vpux {

class AsyncInferRequest final : public ov::IAsyncInferRequest {
public:
    explicit AsyncInferRequest(const std::shared_ptr<SyncInferRequest> inferRequest,
                               const std::shared_ptr<ov::threading::ITaskExecutor> requestExecutor,
                               const std::shared_ptr<ov::threading::ITaskExecutor> getResultExecutor,
                               const std::shared_ptr<ov::threading::ITaskExecutor> callbackExecutor);

    AsyncInferRequest(const AsyncInferRequest&) = delete;

    AsyncInferRequest& operator=(const AsyncInferRequest&) = delete;

    ~AsyncInferRequest();

    std::shared_ptr<SyncInferRequest> get_sync_infer_request() {
        return _syncInferRequest;
    }

private:
    std::shared_ptr<SyncInferRequest> _syncInferRequest;
    std::shared_ptr<ov::threading::ITaskExecutor> _getResultExecutor;
};

}  // namespace vpux
