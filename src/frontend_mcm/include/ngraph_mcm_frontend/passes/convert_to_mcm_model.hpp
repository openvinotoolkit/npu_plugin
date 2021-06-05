//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#pragma once

#include <caseless.hpp>
#include <custom_layer/custom_layer.hpp>
#include <custom_layer/custom_parser_ngraph.hpp>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>
#include <include/mcm/op_model.hpp>
#include <mcm_config.hpp>
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
                      const std::map<std::string, std::string>& ioMap, const vpu::MCMConfig& config,
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
    vpu::MCMConfig _config;
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
