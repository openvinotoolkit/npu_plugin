//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "simulation.hpp"

#include <opencv2/gapi/util/optional.hpp>

#include <string>
#include <vector>

// NB: In order to avoid duplicated tags.
class TagsManager {
public:
    std::string add(const std::string& tag);

private:
    std::unordered_multiset<std::string> m_tags;
};

template <typename K, typename V>
using M = std::unordered_map<K, V>;

class InferenceBuilder {
public:
    struct LayerID {
        std::string tag;
        std::string layer_name;
    };
    using PP = std::function<cv::GMat(cv::GMat)>;
    using OptPP = cv::util::optional<PP>;

    InferenceBuilder();

    void addInference(const std::string& tag, const std::vector<std::string>& in_layer_names,
                      const std::vector<std::string>& out_layer_names);

    void addConnection(const LayerID& src, const LayerID& dst, PP&& pp);

    void addGraphInput(const LayerID& id);
    void addGraphInput(const LayerID& id, PP&& pp);

    void addGraphOutput(const LayerID& id);
    void addGraphOutput(const LayerID& id, PP&& pp);

    using CallBackF = std::function<void(GraphInputs&, GraphOutputs&)>;
    CallBackF build();

    struct Slot {
        LayerID id;
        OptPP pp;
    };

    struct Infer {
        std::string tag;
        std::vector<std::string> in_layer_names;
        std::vector<std::string> out_layer_names;
    };

    struct State {
        M<std::string, M<std::string, Slot>> in_slots_map;
        std::vector<Infer> infers;
        std::vector<Slot> input_slots;
        std::vector<Slot> output_slots;
    };

private:
    std::shared_ptr<State> m_state;
};
