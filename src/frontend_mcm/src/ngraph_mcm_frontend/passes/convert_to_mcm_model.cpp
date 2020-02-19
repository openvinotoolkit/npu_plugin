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

// clang-format off
#ifdef ENABLE_MCM_COMPILER

#include "ngraph_mcm_frontend/passes/convert_to_mcm_model.hpp"
#include "ngraph_mcm_frontend/mcm_attrs.hpp"
#include "ngraph_mcm_frontend/mcm_helpers.hpp"
#include "ngraph_mcm_frontend/ops/mcm_conv.hpp"
#include "ngraph_mcm_frontend/ops/mcm_bias.hpp"
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/result.hpp>
#include <ngraph/op/constant.hpp>
#include <memory>
#include <vector>
#include <map>

namespace {

using Callback = void (*)(std::shared_ptr<ngraph::Node> node, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap);
using DispatchMap = std::map<ngraph::NodeTypeInfo, Callback>;

std::vector<mv::Data::TensorIterator> getMcmInputs(std::shared_ptr<ngraph::Node> node, const NodeOutputToMcmMap& mcmOutputsMap) {
    std::vector<mv::Data::TensorIterator> out;
    out.reserve(node->get_input_size());

    for (const auto& input : node->inputs()) {
        out.push_back(mcmOutputsMap.at(input.get_source_output()));
    }

    return out;
}

void registerOutputs(std::shared_ptr<ngraph::Node> node, std::vector<mv::Data::TensorIterator> mcmOutputs, NodeOutputToMcmMap& mcmOutputsMap) {
    size_t ind = 0;
    for (const auto& mcmOutput : mcmOutputs) {
        mcmOutputsMap.insert({node->output(ind), mcmOutput});
        ++ind;
    }
}

void convert(std::shared_ptr<ngraph::op::Parameter> param, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mvShape = cvtShapeToMCM(param->get_shape());
    const auto mvDType = cvtElemTypeToMCM(param->get_element_type());
    const auto mvOrder = McmOpAttrs::getOrder(param);
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(param);
    const auto& opName = param->get_friendly_name();

    const auto mcmOutput = mcmModel.input(mvShape, mvDType, mvOrder, mvQuantParams, opName);

    registerOutputs(param, {mcmOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<ngraph::op::Result> result, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(result, mcmOutputsMap);
    mcmModel.output(mcmInputs.at(0));
}

void convert(std::shared_ptr<ngraph::op::Constant> constant, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mvShape = cvtShapeToMCM(constant->get_shape());
    const auto mvDType = cvtElemTypeToMCM(constant->get_element_type());
    const auto mvOrder = McmOpAttrs::getOrder(constant);
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(constant);
    const auto& opName = constant->get_friendly_name();

    mv::Data::TensorIterator mcmOutput;
    if (constant->get_element_type().is_real()) {
        mcmOutput = mcmModel.constant(constant->cast_vector<double>(), mvShape, mvDType, mvOrder, mvQuantParams, opName);
    } else {
        mcmOutput = mcmModel.constantInt(constant->cast_vector<int64_t>(), mvShape, mvDType, mvOrder, mvQuantParams, opName);
    }

    registerOutputs(constant, {mcmOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<McmConv> conv, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(conv, mcmOutputsMap);

    const auto mcmData = mcmInputs.at(0);
    const auto mcmWeights = mcmInputs.at(1);

    const auto& strides = conv->get_strides();
    const auto& padsBegin = conv->get_pads_begin();
    const auto& padsEnd = conv->get_pads_end();
    const auto& dilations = conv->get_dilations();
    const auto group = conv->get_group();

    const auto mvDType = cvtElemTypeToMCM(conv->get_element_type());
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(conv);
    const auto& opName = conv->get_friendly_name();

    IE_ASSERT(dilations.at(1) == dilations.at(0));

    const auto mcmConvOutput = mcmModel.conv(
        mcmData, mcmWeights,
        {
            static_cast<uint16_t>(strides.at(1)),
            static_cast<uint16_t>(strides.at(0))},
        {
            static_cast<uint16_t>(padsBegin.at(1)), static_cast<uint16_t>(padsEnd.at(1)),
            static_cast<uint16_t>(padsBegin.at(0)), static_cast<uint16_t>(padsEnd.at(0))
        },
        static_cast<uint32_t>(dilations.at(1)),
        static_cast<uint32_t>(group),
        mvDType, mvQuantParams, opName);

    registerOutputs(conv, {mcmConvOutput}, mcmOutputsMap);
}

void convert(std::shared_ptr<McmBias> bias, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    const auto mcmInputs = getMcmInputs(bias, mcmOutputsMap);

    const auto mcmData = mcmInputs.at(0);
    const auto mcmBias = mcmInputs.at(1);

    const auto mvDType = cvtElemTypeToMCM(bias->get_element_type());
    const auto& mvQuantParams = McmOpAttrs::getQuantParams(bias);
    const auto& opName = bias->get_friendly_name();

    const auto mcmBiasOutput = mcmModel.bias(
        mcmData, mcmBias,
        mvDType, mvQuantParams, opName);

    registerOutputs(bias, {mcmBiasOutput}, mcmOutputsMap);
}

template <typename T>
void convertDispatch(std::shared_ptr<ngraph::Node> node, mv::OpModel& mcmModel, NodeOutputToMcmMap& mcmOutputsMap) {
    convert(std::dynamic_pointer_cast<T>(node), mcmModel, mcmOutputsMap);
}

#define MAP_ENTRY(__OP__) {__OP__::type_info, convertDispatch<__OP__>}

static const DispatchMap dispatchMap {
    MAP_ENTRY(ngraph::op::Parameter),
    MAP_ENTRY(ngraph::op::Result),
    MAP_ENTRY(ngraph::op::Constant),
    MAP_ENTRY(McmConv),
    MAP_ENTRY(McmBias),
};

#undef MAP_ENTRY

}  // namespace

bool ConvertToMcmModel::run_on_function(std::shared_ptr<ngraph::Function> func) {
    for (const auto& op : func->get_ordered_ops()) {
        const auto dispatchIt = dispatchMap.find(op->get_type_info());
        IE_ASSERT(dispatchIt != dispatchMap.end()) << "Unsupported operation " << op->get_friendly_name() << " with type " << op->get_type_name();

        const auto convertor = dispatchIt->second;
        IE_ASSERT(convertor != nullptr);

        convertor(op, _mcmModel, _mcmOutputsMap);
    }

    return false;
}

#endif
// clang-format on
