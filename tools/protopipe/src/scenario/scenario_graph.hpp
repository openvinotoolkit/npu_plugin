//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <functional>

#include <opencv2/gapi/util/variant.hpp>

#include "graph.hpp"
#include "scenario/accuracy_metrics.hpp"
#include "scenario/ovhelper.hpp"
#include "utils/data_providers.hpp"

struct InferDesc {
    std::string tag;
    LayersInfo input_layers;
    LayersInfo output_layers;
    LayerVariantAttr<std::string> input_data;
    LayerVariantAttr<std::string> output_data;
    std::map<std::string, IRandomGenerator::Ptr> generators;
    std::map<std::string, IAccuracyMetric::Ptr> metrics;
};

struct DelayDesc {
    uint64_t delay_in_us;
};

struct OpDesc {
    // NB: variant has explicit ctor.
    template <typename Desc>
    OpDesc(Desc&& desc): kind(std::forward<Desc>(desc)){};

    using Kind = cv::util::variant<InferDesc, DelayDesc>;
    Kind kind;
};

struct SourceDesc {
    uint32_t fps;
};
struct DataDesc {};

class DataNode {
public:
    DataNode(Graph* graph, NodeHandle nh);

private:
    friend class ScenarioGraph;
    NodeHandle m_nh;
};

class OpNode {
public:
    OpNode(NodeHandle nh, DataNode out_data);
    DataNode out(size_t idx = 0);

private:
    friend class ScenarioGraph;
    NodeHandle m_nh;
    std::vector<DataNode> m_out_data;
};

class ScenarioGraph {
public:
    DataNode makeSource(uint32_t target_fps = 0);
    OpNode makeInfer(InferDesc&& desc);
    OpNode makeDelay(uint64_t delay_in_us);

    void link(DataNode data, OpNode op);

    template <typename F>
    void pass(F&& f) {
        f(m_graph);
    }

private:
    OpNode makeOp(OpDesc&& desc);

private:
    Graph m_graph;
};
