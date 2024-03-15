//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <memory>

#include <opencv2/gapi/gproto.hpp>  // cv::GCompileArgs

#include "result.hpp"
#include "scenario/criterion.hpp"
#include "scenario/scenario_graph.hpp"

struct ICompiled {
    using Ptr = std::shared_ptr<ICompiled>;
    virtual Result run(ITermCriterion::Ptr) = 0;
};

struct PipelinedCompiled : public ICompiled {};
struct SyncCompiled : public ICompiled {};

struct ISimulation {
    using Ptr = std::shared_ptr<ISimulation>;

    virtual std::shared_ptr<PipelinedCompiled> compilePipelined(const bool drop_frames, cv::GCompileArgs&& args) = 0;

    virtual std::shared_ptr<SyncCompiled> compileSync(const bool drop_frames, cv::GCompileArgs&& args) = 0;

    virtual ~ISimulation() = default;
};
