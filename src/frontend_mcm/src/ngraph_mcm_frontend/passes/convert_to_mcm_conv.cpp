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

#include "ngraph_mcm_frontend/passes/convert_to_mcm_conv.hpp"
#include "ngraph_mcm_frontend/ops/mcm_conv.hpp"
#include "ngraph_mcm_frontend/ops/mcm_bias.hpp"
#include <details/ie_exception.hpp>
#include <ngraph_ops/convolution_ie.hpp>
#include <memory>

bool ConvertToMcmConv::run_on_node(std::shared_ptr<ngraph::Node> node) {

    if (const auto conv = std::dynamic_pointer_cast<ngraph::op::ConvolutionIE>(node)) {
        const auto mcmConv = std::make_shared<McmConv>(
            conv->input_value(0), conv->input_value(1),
            conv->get_strides(), conv->get_pads_begin(), conv->get_pads_end(),
            conv->get_dilations(), conv->get_output_shape(), conv->get_group(),
            conv->get_output_element_type(0));

        if (conv->get_input_size() == 2) {
            ngraph::replace_node(conv, mcmConv);
        } else {
            IE_ASSERT(conv->get_input_size() == 3);
            const auto mcmBias = std::make_shared<McmBias>(mcmConv, conv->input_value(2), conv->get_output_element_type(0));
            ngraph::replace_node(conv, mcmBias);
        }
        return true;
    }
    return false;
}

#endif
// clang-format on
