//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// clang-format off

#include <ie_common.h>
#include "ngraph_mcm_frontend/passes/convert_to_mcm_fc.hpp"
#include "ngraph_mcm_frontend/ops/mcm_fc.hpp"
#include "ngraph_mcm_frontend/ops/mcm_bias.hpp"

#include <legacy/ngraph_ops/fully_connected.hpp>
#include <ngraph/rt_info.hpp>
#include <memory>

bool ConvertToMcmFC::run_on_node(std::shared_ptr<ngraph::Node> node) {
    if (const auto ngraphFC = std::dynamic_pointer_cast<ngraph::op::FullyConnected>(node)) {
        const auto mcmFC = std::make_shared<McmFC>(
            ngraphFC->input_value(0),
            ngraphFC->input_value(1),
            ngraphFC->get_output_shape(0),
            ngraphFC->get_output_element_type(0));
        ngraph::copy_runtime_info(node, mcmFC);
        if (ngraphFC->get_input_size() == 2) {
            ngraph::replace_node(ngraphFC, mcmFC);
        } else {
            IE_ASSERT(ngraphFC->get_input_size() == 3);
            const auto mcmBias = std::make_shared<McmBias>(mcmFC, ngraphFC->input_value(2), ngraphFC->get_output_element_type(0));
            ngraph::copy_runtime_info(node, mcmBias);
            ngraph::replace_node(ngraphFC, mcmBias);
        }
        return true;
    }
    return false;
}

// clang-format on
