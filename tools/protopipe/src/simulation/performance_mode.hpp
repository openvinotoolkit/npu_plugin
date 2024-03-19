//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <memory>

#include <opencv2/gapi/gproto.hpp>  // cv::GCompileArgs

#include "scenario/scenario_graph.hpp"
#include "simulation/computation.hpp"
#include "simulation/simulation.hpp"

class PerformanceSimulation : public ISimulation {
public:
    struct Options {
        bool inference_only;
        cv::util::optional<uint32_t> target_latency;
    };
    explicit PerformanceSimulation(ScenarioGraph&& graph, const Options& opts);

    std::shared_ptr<PipelinedCompiled> compilePipelined(const bool drop_frames, cv::GCompileArgs&& args) override;

    virtual std::shared_ptr<SyncCompiled> compileSync(const bool drop_frames, cv::GCompileArgs&& args) override;

private:
    Options m_opts;
    Computation m_comp;
};
