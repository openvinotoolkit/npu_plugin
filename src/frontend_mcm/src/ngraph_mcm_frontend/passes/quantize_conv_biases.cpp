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

#include "ngraph_mcm_frontend/passes/quantize_conv_biases.hpp"
#include "ngraph_mcm_frontend/ops/mcm_bias.hpp"
#include "ngraph_mcm_frontend/ops/mcm_conv.hpp"
#include "ngraph_mcm_frontend/mcm_attrs.hpp"
#include "ngraph_mcm_frontend/mcm_helpers.hpp"
#include <details/ie_exception.hpp>
#include <memory>
#include <vector>

namespace {

bool rewrite(ngraph::pattern::Matcher& m) {
    const auto bias = std::dynamic_pointer_cast<McmBias>(m.get_match_root());
    IE_ASSERT(bias != nullptr);

    const auto conv = std::dynamic_pointer_cast<McmConv>(bias->input_value(0).get_node_shared_ptr());
    const auto biases = std::dynamic_pointer_cast<ngraph::op::Constant>(bias->input_value(1).get_node_shared_ptr());
    IE_ASSERT(conv != nullptr && biases != nullptr);

    const auto data = conv->input_value(0).get_node_shared_ptr();
    const auto weights = conv->input_value(1).get_node_shared_ptr();
    IE_ASSERT(data != nullptr && weights != nullptr);

    const auto& dataQuantParams = McmOpAttrs::getQuantParams(data);
    const auto& weightsQuantParams = McmOpAttrs::getQuantParams(weights);

    const auto& dataScales = dataQuantParams.getScale();
    const auto& weightsScales = weightsQuantParams.getScale();

    const auto biasesData = biases->cast_vector<double>();

    // TODO: more accurate code for different shapes

    const bool isDataScalesBroadcasted = dataScales.size() != biasesData.size();
    const bool isWeightsScalesBroadcasted = weightsScales.size() != biasesData.size();
    const bool isBiasScalesBroadcasted = isWeightsScalesBroadcasted && isDataScalesBroadcasted;

    std::vector<double> biasScales(biasesData.size());
    std::vector<int64_t> quantizedBiasesData(biasesData.size());

    //  ZP = 0
    //  ScaleBias = ActivationScale * WeightsScale

    for (size_t i = 0; i < biasesData.size(); i++) {
        const auto activationScale = dataScales[isDataScalesBroadcasted ? 0 : i];
        const auto weightsScale = weightsScales[isWeightsScalesBroadcasted ? 0 : i];
        const auto biasScale = activationScale * weightsScale;

        biasScales[i] = biasScale;
        quantizedBiasesData[i] = static_cast<int64_t>(std::round(biasesData[i] / biasScale));
    }

    if (isBiasScalesBroadcasted) {
        biasScales.resize(1);
    }

    // TODO: other types?
//     conv->setElemType(ngraph::element::u8);

    const auto quantizedBiases = std::make_shared<ngraph::op::Constant>(
        ngraph::element::i64, biases->get_shape(), quantizedBiasesData);
    quantizedBiases->set_friendly_name(biases->get_friendly_name());

    const auto newBiasesQuantParam = makeQuantParams({0}, biasScales);
    McmOpAttrs::setQuantParams(newBiasesQuantParam, quantizedBiases);

    bias->input(1).replace_source_output(quantizedBiases);

    return true;
}

}  // namespace

QuantizeConvBiases::QuantizeConvBiases() {
    const std::vector<double> fakeData(1);

    const auto data = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape {1, 1, 1, 1});
    const auto weights = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape {1, 1, 1, 1});

    const auto conv = std::make_shared<McmConv>(
        data, weights,
        ngraph::Strides {1, 1},
        ngraph::CoordinateDiff {0, 0}, ngraph::CoordinateDiff {0, 0},
        ngraph::Strides {1, 1},
        ngraph::Shape {1},
        1, ngraph::element::f32);

    const auto biases = std::make_shared<ngraph::op::Constant>(
        ngraph::element::f32, ngraph::Shape {1}, fakeData.data());

    const auto bias = std::make_shared<McmBias>(conv, biases, ngraph::element::f32);

    const auto m = std::make_shared<ngraph::pattern::Matcher>(bias);
    add_matcher(m, rewrite, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}

#endif
// clang-format on
