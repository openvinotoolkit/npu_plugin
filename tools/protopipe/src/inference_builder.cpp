//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "inference_builder.hpp"

#include <opencv2/gapi/infer.hpp>

std::string TagsManager::add(const std::string& tag) {
    std::string t = tag;
    m_tags.insert(t);
    const auto c = m_tags.count(t);
    if (c > 1) {
        t += "(" + std::to_string(c) + ")";
    }
    return t;
}

using Slot = InferenceBuilder::Slot;
using PP = InferenceBuilder::PP;
using OptPP = InferenceBuilder::OptPP;

template <typename K, typename V>
using M = std::unordered_map<K, V>;

InferenceBuilder::InferenceBuilder(): m_state(new InferenceBuilder::State{}) {
}

void InferenceBuilder::addInference(const std::string& tag, const std::vector<std::string>& in_layer_names,
                                    const std::vector<std::string>& out_layer_names) {
    m_state->infers.push_back(InferenceBuilder::Infer{tag, in_layer_names, out_layer_names});
}

void InferenceBuilder::addGraphInput(const InferenceBuilder::LayerID& id) {
    Slot slot{id, OptPP{}};
    m_state->input_slots.push_back(slot);
    m_state->in_slots_map[id.tag][id.layer_name] = slot;
}

void InferenceBuilder::addGraphInput(const InferenceBuilder::LayerID& id, PP&& pp) {
    Slot slot{id, OptPP{std::move(pp)}};
    m_state->input_slots.push_back(slot);
    // NB: Preprocessing is needed only before the input but not after
    // so just remove it when add connection.
    // FIXME: Probably don't re-use connections here at all...
    m_state->in_slots_map[id.tag][id.layer_name] = Slot{slot.id, OptPP{}};
}

void InferenceBuilder::addGraphOutput(const InferenceBuilder::LayerID& id, PP&& pp) {
    m_state->output_slots.push_back(Slot{id, OptPP{std::move(pp)}});
};

void InferenceBuilder::addGraphOutput(const InferenceBuilder::LayerID& id) {
    m_state->output_slots.push_back(Slot{id, OptPP{}});
};

void InferenceBuilder::addConnection(const InferenceBuilder::LayerID& src, const InferenceBuilder::LayerID& dst,
                                     PP&& pp) {
    m_state->in_slots_map[dst.tag].emplace(dst.layer_name, Slot{src, OptPP{std::move(pp)}});
}

using S = InferenceBuilder::State;

static void buildCallback(const std::shared_ptr<S>& state, GraphInputs& graph_inputs, GraphOutputs& graph_outputs) {
    using Storage = M<std::string, M<std::string, cv::GMat>>;
    Storage data;

    auto getById = [](const Storage& storage, const InferenceBuilder::LayerID& id) {
        const auto tag_it = storage.find(id.tag);
        if (tag_it == storage.end()) {
            throw std::logic_error("Failed to find input data for: " + id.tag);
        }
        auto&& data = tag_it->second;
        auto g_it = data.find(id.layer_name);
        if (g_it == data.end()) {
            throw std::logic_error("Failed to find data for layer: " + id.layer_name + " of model: " + id.tag);
        }
        return g_it->second;
    };

    for (auto&& slot : state->input_slots) {
        auto&& id = slot.id;
        auto&& pp = slot.pp;
        auto g_in = graph_inputs.create<cv::GMat>();
        data[id.tag].emplace(id.layer_name, pp ? pp.value()(g_in) : g_in);
    }

    for (auto&& infer : state->infers) {
        cv::GInferInputs g_ins;
        auto&& in_slots = state->in_slots_map.at(infer.tag);

        bool has_connection = false;
        for (auto&& name : infer.in_layer_names) {
            const auto it = in_slots.find(name);
            // NB: It's not mandatory that every input layer
            // has connection but for safety reasons make sense
            // to check if at least one layer is connected with other
            // models otherwise it doesn't have any sense to keep
            // this model in graph.
            if (it == in_slots.end()) {
                continue;
            } else {
                has_connection = true;
            }
            auto&& slot = it->second;
            auto g_in = getById(data, slot.id);
            g_ins[name] = slot.pp ? slot.pp.value()(g_in) : g_in;
        }

        if (!has_connection) {
            throw std::logic_error("Model " + infer.tag + " doesn't have connections with other models");
        }

        auto g_outs = cv::gapi::infer(infer.tag, g_ins);
        auto&& out_map = data[infer.tag];
        for (auto&& name : infer.out_layer_names) {
            out_map[name] = g_outs.at(name);
        }
    }

    for (auto&& slot : state->output_slots) {
        auto g_out = getById(data, slot.id);
        graph_outputs.push(slot.pp ? slot.pp.value()(g_out) : g_out);
    }
}

InferenceBuilder::CallBackF InferenceBuilder::build() {
    using namespace std::placeholders;
    return std::bind(buildCallback, m_state, _1, _2);
};
