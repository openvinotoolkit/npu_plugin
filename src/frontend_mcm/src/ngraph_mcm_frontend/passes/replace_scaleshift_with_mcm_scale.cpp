//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// clang-format off

#include "ngraph_mcm_frontend/passes/replace_scaleshift_with_mcm_scale.hpp"

#include <memory>
#include <legacy/ngraph_ops/scaleshift.hpp>

#include "ngraph_mcm_frontend/ops/mcm_bias.hpp"
#include "ngraph_mcm_frontend/ops/mcm_scale.hpp"

#include "ngraph_mcm_frontend/mcm_attrs.hpp"
#include "ngraph_mcm_frontend/mcm_helpers.hpp"

#include <ngraph/op/constant.hpp>

#include "vpux/quantization_helpers.hpp"

#include <ngraph/op/fake_quantize.hpp>


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

// clang-format on
