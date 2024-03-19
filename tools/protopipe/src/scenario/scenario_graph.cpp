//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "scenario/scenario_graph.hpp"

DataNode::DataNode(Graph* graph, NodeHandle nh): m_nh(nh) {
    graph->meta(nh).set(DataDesc{});
};

OpNode::OpNode(NodeHandle nh, DataNode out_data): m_nh(nh), m_out_data({out_data}) {
}

DataNode OpNode::out(size_t idx) {
    ASSERT(m_out_data.size() > idx);
    return m_out_data[idx];
}

DataNode ScenarioGraph::makeSource(uint32_t fps) {
    NodeHandle nh = m_graph.create();
    m_graph.meta(nh).set(SourceDesc{fps});
    return DataNode(&m_graph, nh);
}

void ScenarioGraph::link(DataNode data, OpNode op) {
    m_graph.link(data.m_nh, op.m_nh);
}

OpNode ScenarioGraph::makeInfer(InferDesc&& desc) {
    return makeOp(std::move(desc));
}

OpNode ScenarioGraph::makeDelay(uint64_t delay_in_us) {
    return makeOp(DelayDesc{delay_in_us});
}

OpNode ScenarioGraph::makeOp(OpDesc&& desc) {
    auto op_nh = m_graph.create();
    auto out_nh = m_graph.create();
    m_graph.meta(op_nh).set(std::move(desc));
    m_graph.link(op_nh, out_nh);
    return OpNode(op_nh, DataNode(&m_graph, out_nh));
}
