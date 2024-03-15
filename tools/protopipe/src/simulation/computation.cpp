//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "simulation/computation.hpp"

#include "simulation/operations.hpp"
#include "utils/logger.hpp"

static void buildInfer(Graph* graph, NodeHandle nh, const InferDesc& infer, BuildStrategy::Ptr strategy) {
    auto out_nhs = nh->dstNodes();
    ASSERT(out_nhs.size() == 1);

    auto [providers, in_meta, out_meta, disable_copy] = strategy->build(infer);
    ASSERT(providers.size() == infer.input_layers.size());
    ASSERT(in_meta.size() == infer.input_layers.size());
    ASSERT(out_meta.size() == infer.output_layers.size());

    // FIXME: Perhaps this should be done in separate pass.
    uint64_t delay_in_us = 0u;
    for (auto in_eh : nh->srcEdges()) {
        if (graph->meta(in_eh).has<DelayDesc>()) {
            auto desc = graph->meta(in_eh).get<DelayDesc>();
            delay_in_us = std::max(delay_in_us, desc.delay_in_us);
        }
    }

    auto dummy_nh = graph->create();
    DummyM dummy_op{providers, delay_in_us, disable_copy};
    graph->meta(dummy_nh).set(GOperation{std::move(dummy_op)});
    auto in_nhs = nh->srcNodes();
    for (uint32_t i = 0; i < in_nhs.size(); ++i) {
        graph->meta(graph->link(in_nhs[i], dummy_nh)).set(InputIdx{i});
    }

    graph->remove(nh);

    auto infer_nh = graph->create();
    for (uint32_t layer_idx = 0; layer_idx < infer.input_layers.size(); ++layer_idx) {
        // NB: Create dummy out node and link with dummy.
        auto dummy_out_nh = graph->create();
        graph->meta(dummy_out_nh) += std::move(in_meta[layer_idx]);
        graph->meta(graph->link(dummy_nh, dummy_out_nh)).set(OutputIdx{layer_idx});
        graph->meta(dummy_out_nh).set(GData{});
        // NB: Finally link dummy out with infer
        graph->meta(graph->link(dummy_out_nh, infer_nh)).set(InputIdx{layer_idx});
    }

    auto out_nh = out_nhs.front();
    graph->meta(graph->link(infer_nh, out_nh)).set(OutputIdx{0u});
    graph->meta(out_nh) += out_meta.front();
    for (uint32_t layer_idx = 1; layer_idx < infer.output_layers.size(); ++layer_idx) {
        auto infer_out_nh = graph->create();
        graph->meta(infer_out_nh) = std::move(out_meta[layer_idx]);
        graph->meta(infer_out_nh).set(GData{});
        graph->meta(graph->link(infer_nh, infer_out_nh)).set(OutputIdx{layer_idx});
    }

    Infer infer_op{infer.tag, extractLayerNames(infer.input_layers), extractLayerNames(infer.output_layers)};
    graph->meta(infer_nh).set(GOperation{std::move(infer_op)});
}

static bool fuseDelay(Graph* graph, NodeHandle nh, const DelayDesc& desc) {
    // NB: Access readers of delay output data node.
    auto delay_out_nh = nh->dstNodes().front();
    auto out_edges = delay_out_nh->dstEdges();
    // NB: No delay readers, there is nothing to fuse with.
    if (delay_out_nh->dstNodes().empty()) {
        return false;
    }
    auto in_nhs = nh->srcNodes();
    for (auto out_eh : out_edges) {
        auto reader_nh = out_eh->dstNode();
        auto opdesc = graph->meta(reader_nh).get<OpDesc>().kind;
        if (cv::util::holds_alternative<InferDesc>(opdesc)) {
            // NB: Disconnect from delay readers and connect with writers.
            graph->remove(out_eh);
            for (auto in_nh : in_nhs) {
                graph->meta(graph->link(in_nh, reader_nh)).set(desc);
            }
        }
        // TODO: Can be also fused to another "delay".
    }
    // NB: If there is no delay readers left - just remove it.
    if (delay_out_nh->dstNodes().empty()) {
        graph->remove(nh);
        graph->remove(delay_out_nh);
    }
    return true;
}

static void fuseDelays(Graph& graph) {
    // NB: Iterate over graph nodes until all delays are fused.
    while (true) {
        bool is_fused = false;
        for (auto nh : graph.nodes()) {
            if (!graph.meta(nh).has<OpDesc>()) {
                continue;
            }
            auto opdesc = graph.meta(nh).get<OpDesc>().kind;
            if (cv::util::holds_alternative<DelayDesc>(opdesc)) {
                auto desc = cv::util::get<DelayDesc>(opdesc);
                if (fuseDelay(&graph, nh, desc)) {
                    is_fused = true;
                    break;
                }
            }
        }
        // NB: If delay was fused, some of the nodes were removed
        // Iterate one more time...
        if (!is_fused) {
            break;
        }
    }
};

static void buildDelay(Graph* graph, NodeHandle nh, const DelayDesc& desc) {
    auto in_nhs = nh->srcNodes();
    auto out_nhs = nh->dstNodes();
    graph->remove(nh);

    auto delay_nh = graph->create();
    auto provider = std::make_shared<CircleBuffer>(utils::createRandom({1}, CV_8U));
    DummyM dummy_op{{provider}, desc.delay_in_us};
    graph->meta(delay_nh).set(GOperation{std::move(dummy_op)});

    for (uint32_t i = 0; i < in_nhs.size(); ++i) {
        graph->meta(graph->link(in_nhs[i], delay_nh)).set(InputIdx{i});
    }
    graph->meta(graph->link(delay_nh, out_nhs.front())).set(OutputIdx{0u});
};

static void init(Graph& graph) {
    // NB: Assign i/o contract for the computation graph.
    uint32_t num_sources = 0;
    for (auto nh : graph.nodes()) {
        if (graph.meta(nh).has<SourceDesc>()) {
            ++num_sources;
            graph.meta(nh).set(GraphInput{});
        }
        if (nh->dstNodes().empty()) {
            ASSERT(graph.meta(nh).has<DataDesc>());
            graph.meta(nh).set(GraphOutput{});
        }
        if (!graph.meta(nh).has<OpDesc>()) {
            ASSERT(graph.meta(nh).has<DataDesc>());
            graph.meta(nh).set(GData{});
        }
    }
    ASSERT(num_sources != 0);
};

static void expand(BuildStrategy::Ptr strategy, Graph& graph) {
    for (auto nh : graph.sorted()) {
        if (!graph.meta(nh).has<OpDesc>()) {
            continue;
        }
        auto opdesc = graph.meta(nh).get<OpDesc>().kind;
        switch (opdesc.index()) {
        case OpDesc::Kind::index_of<InferDesc>(): {
            auto desc = cv::util::get<InferDesc>(opdesc);
            buildInfer(&graph, nh, desc, strategy);
            break;
        }
        case OpDesc::Kind::index_of<DelayDesc>(): {
            auto desc = cv::util::get<DelayDesc>(opdesc);
            buildDelay(&graph, nh, desc);
            break;
        }
        default:
            ASSERT(false && "Unsupported operation desc!");
        }
    }

    for (auto nh : graph.nodes()) {
        // NB: Make sure all data nodes that needs to be
        // dumped/validated are graph outputs.
        if (!graph.meta(nh).has<GraphOutput>() && (graph.meta(nh).has<Validate>() || graph.meta(nh).has<Dump>())) {
            graph.meta(nh).set(GraphOutput{});
        }
    }
};

static void buildComputation(Graph& graph, cv::GProtoArgs& graph_inputs, cv::GProtoArgs& graph_outputs,
                             std::vector<SourceDesc>& sources_desc, std::vector<Meta>& out_meta) {
    std::unordered_map<NodeHandle, cv::GProtoArg> all_data;
    auto sorted = graph.sorted();
    for (auto nh : sorted) {
        if (graph.meta(nh).has<GraphInput>()) {
            auto it = all_data.emplace(nh, cv::GProtoArg{cv::GMat()}).first;
            graph_inputs.push_back(it->second);
            ASSERT(graph.meta(nh).has<SourceDesc>());
            sources_desc.push_back(graph.meta(nh).get<SourceDesc>());
        }
    }
    for (auto nh : sorted) {
        ASSERT(graph.meta(nh).has<GOperation>() || graph.meta(nh).has<GData>());
        if (graph.meta(nh).has<GOperation>()) {
            const auto& operation = graph.meta(nh).get<GOperation>();
            // NB: Map input args to the correct input index.
            std::unordered_map<uint32_t, cv::GProtoArg> idx_to_arg;
            auto in_ehs = nh->srcEdges();
            for (auto in_eh : in_ehs) {
                ASSERT(graph.meta(in_eh).has<InputIdx>());
                const uint32_t in_idx = graph.meta(in_eh).get<InputIdx>().idx;
                auto arg = all_data.at(in_eh->srcNode());
                idx_to_arg.emplace(in_idx, arg);
            }
            cv::GProtoArgs in_args;
            for (uint32_t idx = 0; idx < idx_to_arg.size(); ++idx) {
                in_args.push_back(idx_to_arg.at(idx));
            }

            // NB: Link G-API operation with its io data.
            auto out_args = operation.on(in_args);
            // TODO: Validation in/out amount and types...

            // NB: Map output args to the correct index.
            auto out_ehs = nh->dstEdges();
            for (auto out_eh : out_ehs) {
                ASSERT(graph.meta(out_eh).has<OutputIdx>());
                const uint32_t out_idx = graph.meta(out_eh).get<OutputIdx>().idx;
                auto out_nh = out_eh->dstNode();
                all_data.emplace(out_nh, out_args[out_idx]);
            }
        }
    }
    for (auto nh : sorted) {
        if (graph.meta(nh).has<GraphOutput>()) {
            graph_outputs.push_back(all_data.at(nh));
            out_meta.push_back(graph.meta(nh));
        }
    }
}

Computation Computation::build(ScenarioGraph&& graph, BuildStrategy::Ptr&& visitor, const bool add_perf_meta) {
    using namespace std::placeholders;

    // (1) Expand scenario blocks into G-API operations.
    graph.pass(init);
    graph.pass(fuseDelays);
    graph.pass(std::bind(expand, std::move(visitor), _1));

    // (2) Evaluate operations to build cv::GComputation.
    cv::GProtoArgs graph_inputs, graph_outputs;
    std::vector<SourceDesc> sources_desc;
    std::vector<Meta> out_meta;
    using F = std::function<void(Graph & graph)>;
    F f = std::bind(buildComputation, _1, std::ref(graph_inputs), std::ref(graph_outputs), std::ref(sources_desc),
                    std::ref(out_meta));
    graph.pass(f);

    ASSERT(!graph_inputs.empty());
    ASSERT(!graph_outputs.empty());
    // (3) Add performance meta. In fact it should've been done in step (1)...
    if (add_perf_meta) {
        // FIXME: Must work with any G-Type!
        ASSERT(cv::util::holds_alternative<cv::GMat>(graph_outputs.front()));
        cv::GMat g = cv::util::get<cv::GMat>(graph_outputs.front());
        graph_outputs.emplace_back(cv::gapi::streaming::timestamp(g).strip());
        graph_outputs.emplace_back(cv::gapi::streaming::seq_id(g).strip());
    }
    cv::GComputation comp(cv::GProtoInputArgs{std::move(graph_inputs)}, cv::GProtoOutputArgs{std::move(graph_outputs)});
    return {std::move(comp), std::move(sources_desc), std::move(out_meta)};
}

static cv::GMetaArgs descr_of(const std::vector<DummySource::Ptr>& sources) {
    cv::GMetaArgs meta;
    meta.reserve(sources.size());
    for (auto& src : sources) {
        meta.emplace_back(src->descr_of());
    }
    return meta;
}

static std::vector<DummySource::Ptr> createSources(const std::vector<SourceDesc>& sources_desc) {
    std::vector<DummySource::Ptr> sources;
    for (const auto& desc : sources_desc) {
        sources.push_back(std::make_shared<DummySource>(desc.fps));
    }
    return sources;
}

SyncInfo Computation::compileSync(const bool drop_frames, cv::GCompileArgs&& args) {
    auto sources = createSources(m_sources_desc);
    // NB: Copy meta in order to keep invariant.
    auto out_meta = m_out_meta;
    args += cv::compile_args(cv::gapi::kernels<GCPUDummyM>());
    if (drop_frames) {
        for (auto& src : sources) {
            src->enableDropFrames(true);
        }
    }
    auto compiled = m_comp.compile(descr_of(sources), std::move(args));
    return {std::move(compiled), std::move(sources), std::move(out_meta)};
}

PipelinedInfo Computation::compilePipelined(const bool drop_frames, cv::GCompileArgs&& args) {
    auto sources = createSources(m_sources_desc);
    // NB: Copy meta in order to keep invariant.
    auto out_meta = m_out_meta;
    args += cv::compile_args(cv::gapi::kernels<GCPUDummyM>());
    // NB: Hardcoded for pipelining mode as the best option.
    args += cv::compile_args(cv::gapi::streaming::queue_capacity{1u});
    if (drop_frames) {
        THROW_ERROR("Pipelined computation doesn't support frames drop.");
    }
    auto compiled = m_comp.compileStreaming(descr_of(sources), std::move(args));
    return {std::move(compiled), std::move(sources), std::move(out_meta)};
}
