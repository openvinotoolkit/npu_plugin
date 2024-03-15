//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "scenario/scenario_graph.hpp"
#include "simulation/computation.hpp"
#include "simulation/simulation.hpp"

class ReferenceStrategy : public BuildStrategy {
public:
    BuildStrategy::ConstructInfo build(const InferDesc& infer) override;
    // NB: If specified will force execution to perform exactly require_num_iterations
    // regardless what user specified.
    // Use case is when N input iterations are provided,
    // generate exactly the same amount of output iterations.
    // Another use case is when there is only single file provided
    // so only one input / output iteration must be generated.
    cv::optional<uint64_t> required_num_iterations;
};

class CalcRefSimulation : public ISimulation {
public:
    explicit CalcRefSimulation(ScenarioGraph&& graph);

    std::shared_ptr<PipelinedCompiled> compilePipelined(const bool drop_frames, cv::GCompileArgs&& args) override;

    virtual std::shared_ptr<SyncCompiled> compileSync(const bool drop_frames, cv::GCompileArgs&& args) override;

private:
    std::shared_ptr<ReferenceStrategy> m_strategy;
    Computation m_comp;
};
