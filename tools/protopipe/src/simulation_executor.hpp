//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/gapi/streaming/meta.hpp>

#include "criterion.hpp"
#include "utils.hpp"

class SimulationExecutor {
public:
    using Ptr = std::shared_ptr<SimulationExecutor>;

    struct Output {
        std::vector<int64_t> latency;
        std::vector<int64_t> seq_ids;
        int64_t elapsed;
    };

    virtual void setSource(cv::GRunArgs&& sources);
    virtual Output runLoop(ITermCriterion::Ptr criterion) = 0;

    virtual void runWarmup(){/* do nothing (default) */};
    virtual void postIterationCallback() { /* do nothing (default) */
    }
    virtual void validate(){/* do nothing (default) */};

    virtual ~SimulationExecutor() = default;

protected:
    cv::GRunArgs m_pipeline_inputs;
};

class SyncExecutor : public SimulationExecutor {
public:
    using Ptr = std::shared_ptr<SyncExecutor>;
    SyncExecutor(cv::GCompiled&& compiled);

    SimulationExecutor::Output runLoop(ITermCriterion::Ptr criterion) override;

    void runWarmup() override;
    void setSource(cv::GRunArgs&& sources) override;
    void setDropFrames(const bool drop_frames) {
        m_drop_frames = drop_frames;
    };

protected:
    virtual cv::GRunArgsP outputs() = 0;

private:
    cv::GRunArgs fetchInputs(cv::GRunArgs&& inputs);

protected:
    cv::GCompiled m_compiled;
    std::vector<int> m_src_ids;
    bool m_drop_frames;
};

class PipelinedExecutor : public SimulationExecutor {
public:
    using Ptr = std::shared_ptr<PipelinedExecutor>;
    PipelinedExecutor(cv::GStreamingCompiled&& stream);

    SimulationExecutor::Output runLoop(ITermCriterion::Ptr criterion) override;

    void runWarmup() override;

protected:
    virtual cv::GOptRunArgsP outputs() = 0;

    cv::GStreamingCompiled m_stream;
};
