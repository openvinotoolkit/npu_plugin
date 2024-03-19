//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <mutex>

#include "vpux.hpp"
#include "vpux/utils/core/logger.hpp"
#include "zero_executor.h"
#include "zero_pipeline.h"
#include "zero_profiling.h"
#include "zero_utils.h"

#include <ze_api.h>
#include <ze_graph_ext.h>

namespace vpux {

class ZeroInferRequest final : public SyncInferRequest {
public:
    using Ptr = std::shared_ptr<ZeroInferRequest>;

    explicit ZeroInferRequest(const std::shared_ptr<const ov::ICompiledModel> compiledModel,
                              const std::shared_ptr<const NetworkDescription> networkDescription,
                              const Executor::Ptr executor, const Config& config);

    void infer() override;
    void infer_async() override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    void get_result() override;

private:
    void check_network_precision(const ov::element::Type_t precision) override;

    const Executor::Ptr _executorPtr;
    const ZeroExecutor* _executor;
    const Config _config;
    Logger _logger;

    vpux::zeroProfiling::ProfilingPool _profiling_pool;
    vpux::zeroProfiling::ProfilingQuery _profiling_query;
    std::shared_ptr<vpux::zeroProfiling::VpuInferProfiling> _vpu_profiling = nullptr;
    std::unique_ptr<Pipeline> _pipeline;
};

}  //  namespace vpux
