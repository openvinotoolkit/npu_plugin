//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <filesystem>

#include "scenario/scenario_graph.hpp"
#include "simulation/computation.hpp"
#include "simulation/simulation.hpp"

class ValSimulation : public ISimulation {
public:
    struct Options {
        cv::util::optional<std::filesystem::path> per_iter_outputs_path;
    };
    explicit ValSimulation(ScenarioGraph&& graph, const Options& options);

    std::shared_ptr<PipelinedCompiled> compilePipelined(const bool drop_frames, cv::GCompileArgs&& args) override;

    virtual std::shared_ptr<SyncCompiled> compileSync(const bool drop_frames, cv::GCompileArgs&& args) override;

private:
    Options m_opts;
    Computation m_comp;
};
