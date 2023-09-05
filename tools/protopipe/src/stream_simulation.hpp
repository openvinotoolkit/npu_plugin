//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "simulation.hpp"

#include <unordered_map>

struct ReferenceInfo {
    // NB: IF ref_data isn't provided - compare with the first run.
    cv::optional<cv::Mat> ref_data;
    std::string model_name;
    std::string layer_name;
};

// NB: output idx -> ReferenceInfo.
using ValidationMap = std::unordered_map<size_t, ReferenceInfo>;

struct ValidationInfo {
    bool save_per_iter_outputs;
    // NB: Only for creating dump dir name.
    size_t stream_idx;
    ValidationMap map;
};

class StreamSimulation : public Simulation {
public:
    StreamSimulation(Simulation::GraphBuildF&& build, const size_t num_outputs, ValidationInfo&& validation_info);

    SyncExecutor::Ptr compileSync(cv::GCompiled&& sync) override;

    PipelinedExecutor::Ptr compilePipelined(cv::GStreamingCompiled&& pipelined) override;

private:
    size_t m_num_outputs;
    ValidationInfo m_validation_info;
};
