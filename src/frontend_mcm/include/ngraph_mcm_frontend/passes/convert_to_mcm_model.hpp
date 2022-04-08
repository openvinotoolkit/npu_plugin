//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "vpux/utils/IE/config.hpp"

#include <caseless.hpp>
#include <custom_layer/custom_layer.hpp>
#include <custom_layer/custom_parser_ngraph.hpp>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>
#include <include/mcm/op_model.hpp>
#include <memory>
#include <ngraph/pass/pass.hpp>

#include "ngraph_mcm_frontend/mcm_helpers.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

//
// Convert nGraph Function to MCM OpModel.
//

class ConvertToMcmModel final : public ngraph::pass::FunctionPass {
public:
    ConvertToMcmModel(mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
                      const InferenceEngine::InputsDataMap& networkInputs,
                      const InferenceEngine::OutputsDataMap& networkOutputs,
                      const std::map<std::string, std::string>& ioMap, const vpux::Config& config,
                      bool* needConvertInputPrecision)
            : _mcmModel(mcmModel),
              _mcmOutputsMap(mcmOutputsMap),
              _networkInputs(networkInputs),
              _networkOutputs(networkOutputs),
              _ioMap(ioMap),
              _config(config),
              _needConvertInputPrecision(needConvertInputPrecision) {
    }

    bool run_on_function(std::shared_ptr<ngraph::Function> func) override;

    void parseCustom(std::shared_ptr<ngraph::Node> node, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap);

    InferenceEngine::details::caseless_map<std::string, std::vector<vpu::CustomLayer::Ptr>> _customLayers;

private:
    mv::OpModel& _mcmModel;
    NodeOutputToMcmMap& _mcmOutputsMap;
    InferenceEngine::InputsDataMap _networkInputs;
    InferenceEngine::OutputsDataMap _networkOutputs;
    std::map<std::string, std::string> _ioMap;
    const vpux::Config _config;
    bool* _needConvertInputPrecision;
};

class QueryModel final : public ngraph::pass::FunctionPass {
public:
    QueryModel(std::shared_ptr<std::unordered_set<std::string>> supported): _supported(supported) {
    }
    bool run_on_function(std::shared_ptr<ngraph::Function> func) override;

private:
    std::shared_ptr<std::unordered_set<std::string>> _supported;
};
