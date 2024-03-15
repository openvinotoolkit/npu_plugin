//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <filesystem>

#include "result.hpp"
#include "scenario/scenario_graph.hpp"
#include "simulation/dummy_source.hpp"

#include <opencv2/gapi/streaming/meta.hpp>

struct InputIdx {
    uint32_t idx;
};
struct OutputIdx {
    uint32_t idx;
};
struct GraphInput {};
struct GraphOutput {};
struct GData {};
struct GOperation {
    using F = std::function<cv::GProtoArgs(const cv::GProtoArgs&)>;
    F on;
};

struct Dump {
    std::filesystem::path path;
};

struct Validate {
    using F = std::function<Result(const cv::Mat& lhs, const cv::Mat& rhs)>;
    F validator;
    std::vector<cv::Mat> reference;
};

struct BuildStrategy {
    using Ptr = std::shared_ptr<BuildStrategy>;
    struct ConstructInfo {
        std::vector<IDataProvider::Ptr> providers;
        std::vector<Meta> inputs_meta;
        std::vector<Meta> outputs_meta;
        const bool disable_copy;
    };
    virtual ConstructInfo build(const InferDesc& infer) = 0;
    virtual ~BuildStrategy() = default;
};

struct SyncInfo {
    cv::GCompiled compiled;
    std::vector<DummySource::Ptr> sources;
    std::vector<Meta> out_meta;
};

struct PipelinedInfo {
    cv::GStreamingCompiled compiled;
    std::vector<DummySource::Ptr> sources;
    std::vector<Meta> out_meta;
};

class Computation {
public:
    static Computation build(ScenarioGraph&& graph, BuildStrategy::Ptr&& strategy, const bool add_perf_meta);

    Computation(cv::GComputation&& comp, std::vector<SourceDesc>&& sources_desc, std::vector<Meta>&& out_meta)
            : m_comp(std::move(comp)), m_sources_desc(std::move(sources_desc)), m_out_meta(std::move(out_meta)) {
    }

    SyncInfo compileSync(const bool drop_frames, cv::GCompileArgs&& args);
    PipelinedInfo compilePipelined(const bool drop_frames, cv::GCompileArgs&& args);

private:
    cv::GComputation m_comp;
    std::vector<SourceDesc> m_sources_desc;
    std::vector<Meta> m_out_meta;
};
