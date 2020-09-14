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

#include <ngraph/op/fake_quantize.hpp>


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
        const auto scales = std::dynamic_pointer_cast<ngraph::op::Constant>(ieScale->input_value(1).get_node_shared_ptr());
        if (!scales)
            return false;
        const auto scalesData = scales->cast_vector<double>();
        const auto reshapedScales = std::make_shared<ngraph::op::Constant>(
            scales->get_element_type(),
            ngraph::Shape({scales->get_shape().at(1)}),
            scalesData);
        reshapedScales->set_friendly_name(scales->get_friendly_name());
        ngraph::replace_node(scales, reshapedScales);

        const auto shifts = std::dynamic_pointer_cast<ngraph::op::Constant>(ieScale->input_value(2).get_node_shared_ptr());
        if (!shifts)
            return false;

        const auto shiftsData = shifts->cast_vector<double>();
        IE_ASSERT(shiftsData.size() == scalesData.size());

        const auto reshapedShifts = std::make_shared<ngraph::op::Constant>(
            shifts->get_element_type(),
            ngraph::Shape({shifts->get_shape().at(1)}),
            shiftsData);
        reshapedShifts->set_friendly_name(shifts->get_friendly_name());
        ngraph::replace_node(shifts, reshapedShifts);

        const auto mcmScale = std::make_shared<McmScale>(
            ieScale->input_value(0), reshapedScales, ieScale->get_output_element_type(0));

        const auto mcmBias = std::make_shared<McmBias>(
            mcmScale, reshapedShifts, mcmScale->get_output_element_type(0));

        ngraph::replace_node(ieScale, mcmBias);

        return true;
    }

    return false;
}

#endif  // ENABLE_MCM_COMPILER
// clang-format on
