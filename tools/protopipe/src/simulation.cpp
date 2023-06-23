//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simulation.hpp"
#include "simulation_executor.hpp"

cv::GProtoArgs GraphInputs::produce() {
    if (m_inputs.empty()) {
        throw std::logic_error("GraphInputs is empty");
    }
    return m_inputs;
}

cv::GProtoArgs GraphOutputs::produce() {
    if (m_outputs.empty()) {
        throw std::logic_error("GraphOutputs is empty");
    }
    cv::GProtoArgs outputs = m_outputs;
    // FIXME: Must work with any G-type.
    GAPI_Assert(cv::util::holds_alternative<cv::GMat>(outputs[0]));
    cv::GMat g = cv::util::get<cv::GMat>(outputs[0]);
    outputs.emplace_back(cv::gapi::streaming::timestamp(g).strip());
    outputs.emplace_back(cv::gapi::streaming::seq_id(g).strip());
    return outputs;
}

Simulation::Simulation(GraphBuildF&& build): m_build(std::move(build)) {
}

SyncExecutor::Ptr Simulation::compileSync(const bool drop_frames, cv::GMetaArgs&& meta, cv::GCompileArgs&& args) {
    GraphInputs graph_inputs;
    GraphOutputs graph_outputs;

    m_build(graph_inputs, graph_outputs);

    cv::GComputation comp(cv::GProtoInputArgs{graph_inputs.produce()}, cv::GProtoOutputArgs{graph_outputs.produce()});

    auto sync = compileSync(comp.compile(std::move(meta), std::move(args)));

    // NB: In case sync simulation frame drop should
    // be implemented on the source level.
    sync->setDropFrames(drop_frames);
    return sync;
}

PipelinedExecutor::Ptr Simulation::compilePipelined(const bool drop_frames, cv::GMetaArgs&& meta,
                                                    cv::GCompileArgs&& args) {
    // NB: In case streaming simulation frame drop
    // hasn't been defined yet ...
    if (drop_frames) {
        throw std::logic_error("Pipelined simulation doesn't support frame drop");
    }

    GraphInputs graph_inputs;
    GraphOutputs graph_outputs;

    m_build(graph_inputs, graph_outputs);

    cv::GComputation comp(cv::GProtoInputArgs{graph_inputs.produce()}, cv::GProtoOutputArgs{graph_outputs.produce()});

    return compilePipelined(comp.compileStreaming(std::move(meta), std::move(args)));
}
