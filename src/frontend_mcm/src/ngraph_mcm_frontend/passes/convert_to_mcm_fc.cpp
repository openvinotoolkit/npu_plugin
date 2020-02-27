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

#include "ngraph_mcm_frontend/passes/convert_to_mcm_fc.hpp"
#include "ngraph_mcm_frontend/ops/mcm_fc.hpp"
#include "ngraph_mcm_frontend/ops/mcm_bias.hpp"

#include <details/ie_exception.hpp>
#include <ngraph_ops/fully_connected.hpp>
#include <memory>

bool ConvertToMcmFC::run_on_node(std::shared_ptr<ngraph::Node> node) {
    if (const auto ngraphFC = std::dynamic_pointer_cast<ngraph::op::FullyConnected>(node)) {
        const auto mcmFC = std::make_shared<McmFC>(
            ngraphFC->input_value(0),
            ngraphFC->input_value(1),
            ngraphFC->get_output_shape(0),
            ngraphFC->get_output_element_type(0));

        if (ngraphFC->get_input_size() == 2) {
            ngraph::replace_node(ngraphFC, mcmFC);
        } else {
            IE_ASSERT(ngraphFC->get_input_size() == 3);
            const auto mcmBias = std::make_shared<McmBias>(mcmFC, ngraphFC->input_value(2), ngraphFC->get_output_element_type(0));
            ngraph::replace_node(ngraphFC, mcmBias);
        }
        return true;
    }
    return false;
}

#endif
// clang-format on
