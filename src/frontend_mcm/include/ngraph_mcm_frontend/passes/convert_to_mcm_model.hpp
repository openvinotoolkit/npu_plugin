//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#pragma once

#include <caseless.hpp>
#include <custom_layer/custom_layer.hpp>
#include <custom_layer/custom_parser_ngraph.hpp>
#include <ie_icnn_network.hpp>
#include <ie_input_info.hpp>
#include <include/mcm/op_model.hpp>
#include <memory>
#include <ngraph/pass/pass.hpp>

#include "ngraph_mcm_frontend/mcm_helpers.hpp"

//
// Convert nGraph Function to MCM OpModel.
//

class ConvertToMcmModel final : public ngraph::pass::FunctionPass {
public:
    ConvertToMcmModel(mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap,
        const InferenceEngine::InputsDataMap& networkInputs, const InferenceEngine::OutputsDataMap& networkOutputs,
        const std::map<std::string, std::string>& ioMap, const vpu::MCMConfig& config)
        : _mcmModel(mcmModel),
          _mcmOutputsMap(mcmOutputsMap),
          _networkInputs(networkInputs),
          _networkOutputs(networkOutputs),
          _ioMap(ioMap),
          _config(config) {}

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
};
