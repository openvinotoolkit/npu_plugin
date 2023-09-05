//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "simulation_executor.hpp"

#include <opencv2/gapi/gproto.hpp>

// NB: Tiny mediator over graph inputs in order to
// apply some transformations on inputs (e.g desync)
class GraphInputs {
public:
    template <typename G>
    G create();
    cv::GProtoArgs produce();

private:
    cv::GProtoArgs m_inputs;
};

// NB: Tiny mediator over graph outputs in order to
// apply some transformations on
// outputs (e.g timestamps, seq_id etc)
class GraphOutputs {
public:
    template <typename G>
    void push(G&& obj);
    cv::GProtoArgs produce();

private:
    cv::GProtoArgs m_outputs;
};

class Simulation {
public:
    using Ptr = std::shared_ptr<Simulation>;
    using GraphBuildF = std::function<void(GraphInputs&, GraphOutputs&)>;

    Simulation(GraphBuildF&& build);

    virtual SyncExecutor::Ptr compileSync(const bool drop_frames, cv::GMetaArgs&& meta, cv::GCompileArgs&& args);

    virtual PipelinedExecutor::Ptr compilePipelined(const bool drop_frames, cv::GMetaArgs&& meta,
                                                    cv::GCompileArgs&& args);

    virtual ~Simulation() = default;

protected:
    virtual SyncExecutor::Ptr compileSync(cv::GCompiled&& sync) = 0;

    virtual PipelinedExecutor::Ptr compilePipelined(cv::GStreamingCompiled&& pipelined) = 0;

private:
    GraphBuildF m_build;
};

template <typename G>
G GraphInputs::create() {
    auto in = G{};
    m_inputs.emplace_back(in);
    return in;
};

template <typename G>
void GraphOutputs::push(G&& obj) {
    m_outputs.emplace_back(obj);
}
