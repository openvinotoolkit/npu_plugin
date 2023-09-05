//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simulation.hpp"

// FIXME: Ideally simulation shouldn't know about specific type of source.
#include "dummy_source.hpp"

#include <opencv2/gapi/gcompiled.hpp>
#include <opencv2/gapi/gstreaming.hpp>

#include <chrono>

static cv::GRunArgs copyInputs(const cv::GRunArgs& inputs) {
    cv::GRunArgs pipeline_inputs = inputs;
    int idx = 0;
    for (auto&& in : pipeline_inputs) {
        using S = cv::gapi::wip::IStreamSource::Ptr;
        if (cv::util::holds_alternative<S>(in)) {
            // FIXME: Source could change the state after pull is triggered and
            // it could affect the computation after warmup.
            // Probable source could implement IResettable or ICloneable interface...
            auto dummy = std::dynamic_pointer_cast<DummySource>(cv::util::get<S>(in));
            if (!dummy) {
                throw std::logic_error("SimulationExecutor supports warmup only for dummy sources");
            }
            S copy = std::make_shared<DummySource>(*dummy);
            pipeline_inputs[idx] = std::move(copy);
        }
        ++idx;
    }
    return pipeline_inputs;
}

void SimulationExecutor::setSource(cv::GRunArgs&& ins) {
    m_pipeline_inputs = std::move(ins);
}

SyncExecutor::SyncExecutor(cv::GCompiled&& compiled): m_compiled(std::move(compiled)) {
}

void SyncExecutor::setSource(cv::GRunArgs&& ins) {
    m_pipeline_inputs = std::move(ins);
    using S = cv::gapi::wip::IStreamSource::Ptr;
    for (auto&& in : m_pipeline_inputs) {
        if (cv::util::holds_alternative<S>(in)) {
            auto src = cv::util::get<S>(in);
            if (m_drop_frames) {
                // FIXME: Ideally it should be wrapped into drop frames decorator
                // so that it works with any type of source.
                auto dummy = std::dynamic_pointer_cast<DummySource>(src);
                if (!dummy) {
                    throw std::logic_error("SyncExecutor supports drop frames only for dummy source.");
                }
                dummy->setDropFrames(true);
            }
        }
    }
};

void SyncExecutor::runWarmup() {
    auto copy_inputs = copyInputs(m_pipeline_inputs);
    auto pipeline_inputs = fetchInputs(std::move(copy_inputs));
    auto pipeline_outputs = outputs();

    int64_t ts = -1;
    int64_t seq_id = -1;
    pipeline_outputs += cv::gout(ts, seq_id);

    m_compiled(std::move(pipeline_inputs), std::move(pipeline_outputs));
}

cv::GRunArgs SyncExecutor::fetchInputs(cv::GRunArgs&& inputs) {
    cv::GRunArgs pipeline_inputs;
    for (auto&& in : inputs) {
        using S = cv::gapi::wip::IStreamSource::Ptr;
        if (cv::util::holds_alternative<S>(in)) {
            auto src = cv::util::get<S>(in);
            cv::gapi::wip::Data d;
            if (!src->pull(d)) {
                throw std::logic_error("SyncExecutor failed to pull input data!");
            }
            pipeline_inputs.push_back(std::move(d));
        } else {
            pipeline_inputs.push_back(std::move(in));
        }
    }

    return pipeline_inputs;
}

SimulationExecutor::Output SyncExecutor::runLoop(ITermCriterion::Ptr criterion) {
    using namespace std::chrono;
    using clock_t = high_resolution_clock;

    SimulationExecutor::Output simout;
    int64_t ts = -1;
    int64_t seq_id = -1;

    criterion->init();

    auto start = clock_t::now();
    while (criterion->check()) {
        auto pipeline_inputs = fetchInputs(cv::GRunArgs{m_pipeline_inputs});
        auto pipeline_outputs = outputs();
        pipeline_outputs += cv::gout(ts, seq_id);

        m_compiled(std::move(pipeline_inputs), std::move(pipeline_outputs));

        simout.latency.push_back(utils::timestamp<microseconds>() - ts);
        simout.seq_ids.push_back(seq_id);

        postIterationCallback();
        criterion->update();
    }

    simout.elapsed = duration_cast<microseconds>(clock_t::now() - start).count();
    return simout;
};

PipelinedExecutor::PipelinedExecutor(cv::GStreamingCompiled&& stream): m_stream(std::move(stream)) {
}

void PipelinedExecutor::runWarmup() {
    cv::optional<int64_t> ts, seq_id;
    auto pipeline_outputs = outputs();
    pipeline_outputs.emplace_back(cv::gout(ts)[0]);
    pipeline_outputs.emplace_back(cv::gout(seq_id)[0]);

    m_stream.setSource(copyInputs(m_pipeline_inputs));
    m_stream.start();

    if (!m_stream.pull(cv::GOptRunArgsP{pipeline_outputs})) {
        // FIXME: Need to handle early stop somehow...
        throw std::logic_error("PipelinedExecutor failed to pull input data!");
    }
    m_stream.stop();
}

SimulationExecutor::Output PipelinedExecutor::runLoop(ITermCriterion::Ptr criterion) {
    using namespace std::chrono;
    using clock_t = high_resolution_clock;

    SimulationExecutor::Output simout;
    cv::optional<int64_t> ts, seq_id;

    m_stream.setSource(cv::GRunArgs{m_pipeline_inputs});
    m_stream.start();

    criterion->init();

    auto start = clock_t::now();
    while (criterion->check()) {
        auto pipeline_outputs = outputs();
        // FIXME: No cv::GOptRunAgsP::operator+=
        pipeline_outputs.emplace_back(cv::gout(ts)[0]);
        pipeline_outputs.emplace_back(cv::gout(seq_id)[0]);

        if (!m_stream.pull(cv::GOptRunArgsP{pipeline_outputs})) {
            // FIXME: Need to handle early stop somehow...
            throw std::logic_error("PipelinedExecutor failed to pull input data!");
        }

        if (!ts.has_value()) {
            throw std::logic_error("PipelinedExecutor failed to obtain timestamp!");
        }
        simout.latency.push_back(utils::timestamp<microseconds>() - ts.value());

        if (!seq_id.has_value()) {
            throw std::logic_error("PipelinedExecutor failed to obtain timestamp!");
        }
        simout.seq_ids.push_back(seq_id.value());

        postIterationCallback();
        criterion->update();
    }
    m_stream.stop();

    simout.elapsed = duration_cast<microseconds>(clock_t::now() - start).count();
    return simout;
}
