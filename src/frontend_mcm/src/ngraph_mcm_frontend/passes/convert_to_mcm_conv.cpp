//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// clang-format off
#include <ie_common.h>
#include "ngraph_mcm_frontend/passes/convert_to_mcm_conv.hpp"
#include "ngraph_mcm_frontend/ops/mcm_conv.hpp"
#include "ngraph_mcm_frontend/ops/mcm_bias.hpp"
#include <legacy/ngraph_ops/convolution_ie.hpp>
#include <ngraph/rt_info.hpp>
#include <memory>

bool ConvertToMcmConv::run_on_node(std::shared_ptr<ngraph::Node> node) {

    if (const auto conv = std::dynamic_pointer_cast<ngraph::op::ConvolutionIE>(node)) {
        const auto mcmConv = std::make_shared<McmConv>(
            conv->input_value(0), conv->input_value(1),
            conv->get_strides(), conv->get_pads_begin(), conv->get_pads_end(),
            conv->get_dilations(), conv->get_group(),
            conv->get_output_element_type(0));

        mcmConv->set_friendly_name(conv->get_friendly_name());
        ngraph::copy_runtime_info(node, mcmConv);

        if (conv->get_input_size() == 2) {
            ngraph::replace_node(conv, mcmConv);
        } else {
            IE_ASSERT(conv->get_input_size() == 3);
            const auto mcmBias = std::make_shared<McmBias>(mcmConv, conv->input_value(2), conv->get_output_element_type(0));
            ngraph::copy_runtime_info(node, mcmBias);
            ngraph::replace_node(conv, mcmBias);
        }
        return true;
    }
    return false;
}

// clang-format on
