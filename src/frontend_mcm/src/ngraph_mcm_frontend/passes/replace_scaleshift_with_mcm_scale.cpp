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

#include "ngraph_mcm_frontend/passes/replace_scaleshift_with_mcm_scale.hpp"

#include <details/ie_exception.hpp>
#include <memory>
//#include <ngraph/pass/pass.hpp>
#include <ngraph_ops/scaleshift.hpp>

#include "ngraph_mcm_frontend/ops/mcm_bias.hpp"
#include "ngraph_mcm_frontend/ops/mcm_scale.hpp"

#include "ngraph_mcm_frontend/mcm_attrs.hpp"
#include "ngraph_mcm_frontend/mcm_helpers.hpp"

#include <ngraph/op/constant.hpp>

#include "ngraph_mcm_frontend/quantization_helpers.hpp"

#include <ngraph/op/fused/fake_quantize.hpp>


mv::QuantizationParams calcQuantParams(const float outputLowMin, const float outputHighMax, const int levels)
{
    std::vector<int64_t> zeroPoints;
    std::vector<double> scales;
    std::vector<double> mins;
    std::vector<double> maxs;

    int64_t zepoPoint = calcZeroPoint(outputLowMin, outputHighMax, levels, ngraph::element::u8); 

    zeroPoints.push_back(static_cast<int64_t>(zepoPoint));
    scales.push_back(static_cast<double>((outputHighMax - outputLowMin) / (levels - 1)));
    mins.push_back(outputLowMin);
    maxs.push_back(outputHighMax);

    return mv::QuantizationParams({zeroPoints, scales, mins, maxs});
}

bool ReplaceScaleShiftWithMcmScale::run_on_node(std::shared_ptr<ngraph::Node> node)
{
    if (const auto ieScale = std::dynamic_pointer_cast<ngraph::op::ScaleShiftIE>(node))
    {
        const auto ieScaleOutputs = ieScale->output(0).get_target_inputs();
        IE_ASSERT(1 == ieScaleOutputs.size());
        const auto outputNode = ieScaleOutputs.begin()->get_node();
        const auto fq =  dynamic_cast<ngraph::op::FakeQuantize*>(outputNode);

        const auto inputLow = std::dynamic_pointer_cast<ngraph::op::Constant>(fq->input_value(1).get_node_shared_ptr());
        const auto inputHigh = std::dynamic_pointer_cast<ngraph::op::Constant>(fq->input_value(2).get_node_shared_ptr());
        const auto outputLow = std::dynamic_pointer_cast<ngraph::op::Constant>(fq->input_value(3).get_node_shared_ptr());
        const auto outputHigh = std::dynamic_pointer_cast<ngraph::op::Constant>(fq->input_value(4).get_node_shared_ptr());
        IE_ASSERT(inputLow != nullptr && inputHigh != nullptr && outputLow != nullptr && outputHigh != nullptr);
        IE_ASSERT(inputLow->get_shape() == inputHigh->get_shape());
        IE_ASSERT(outputLow->get_shape() == outputHigh->get_shape());

        const auto levels = fq->get_levels();
        const auto inputLowData = inputLow->cast_vector<double>();
        const auto inputHighData = inputHigh->cast_vector<double>();
        const auto outputLowData = outputLow->cast_vector<double>();
        const auto outputHighData = outputHigh->cast_vector<double>();

        const auto scales = std::dynamic_pointer_cast<ngraph::op::Constant>(ieScale->input_value(1).get_node_shared_ptr());
        IE_ASSERT(ngraph::element::f32 == scales->get_element_type());
        const auto scalesData = scales->cast_vector<double>();
        std::vector<uint8_t> quantScalesData(scalesData.size(), 1);
        const auto reshapedScales = std::make_shared<ngraph::op::Constant>(
            ngraph::element::u8, // scales->get_element_type(),
            ngraph::Shape({scales->get_shape().at(1)}),
            quantScalesData); // scalesData);
        reshapedScales->set_friendly_name(scales->get_friendly_name());
        ngraph::replace_node(scales, reshapedScales);
        const double inf = std::numeric_limits<double>::infinity();
        const auto newScalesQuantParam = mv::QuantizationParams({0}, scalesData, {-inf}, {inf}); //makeQuantParams({0}, scalesData);
        McmOpAttrs::setQuantParams(newScalesQuantParam, reshapedScales);

        const auto shifts = std::dynamic_pointer_cast<ngraph::op::Constant>(ieScale->input_value(2).get_node_shared_ptr());
        IE_ASSERT(ngraph::element::f32 == shifts->get_element_type());

        const auto shiftsData = shifts->cast_vector<double>();
        std::vector<int32_t> quantizedBiasData(shiftsData.size(), 0);

        IE_ASSERT(shiftsData.size() == scalesData.size());
        for (size_t i = 0; i < shiftsData.size(); i++) {
            quantizedBiasData[i] = std::round(shiftsData[i] / scalesData[i]);
        }

        const auto reshapedShifts = std::make_shared<ngraph::op::Constant>(
            ngraph::element::i32, // scales->get_element_type(),
            ngraph::Shape({shifts->get_shape().at(1)}),
            quantizedBiasData); //shiftsData);
        reshapedShifts->set_friendly_name(shifts->get_friendly_name());
        ngraph::replace_node(shifts, reshapedShifts);

        const std::vector<int64_t> zp(scalesData.size(), 0);
        const auto newBiasQuantParam =  mv::QuantizationParams({0}, scalesData, {0.0}, {1.0});
        McmOpAttrs::setQuantParams(newBiasQuantParam, reshapedShifts);

        // const auto mcmScalesQuantParam = mv::QuantizationParams({128},{0.0197619},{-2.52952},{2.50976}); // TODO Compute
        const auto mcmScalesQuantParam = calcQuantParams(outputLowData.front(), outputHighData.front(), levels);
        const auto mcmBiasQuantParam = mcmScalesQuantParam;

        const auto mcmScale = std::make_shared<McmScale>(
            ieScale->input_value(0), reshapedScales, ieScale->get_output_element_type(0));
       McmOpAttrs::setQuantParams(mcmScalesQuantParam, mcmScale);

        const auto mcmBias = std::make_shared<McmBias>(
            mcmScale, reshapedShifts, mcmScale->get_output_element_type(0));
       McmOpAttrs::setQuantParams(mcmBiasQuantParam, mcmBias);

        ngraph::replace_node(ieScale, mcmBias);

        return true;
    }

    return false;
}

#endif  // ENABLE_MCM_COMPILER
// clang-format on
