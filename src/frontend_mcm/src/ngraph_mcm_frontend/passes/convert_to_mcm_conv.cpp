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

// clang-format off
#include <ie_common.h>
#include "ngraph_mcm_frontend/passes/convert_to_mcm_conv.hpp"
#include "ngraph_mcm_frontend/ops/mcm_conv.hpp"
#include "ngraph_mcm_frontend/ops/mcm_bias.hpp"
#include <ngraph_ops/convolution_ie.hpp>
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
