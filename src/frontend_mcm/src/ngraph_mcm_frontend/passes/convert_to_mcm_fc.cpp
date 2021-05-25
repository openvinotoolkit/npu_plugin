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
#include "ngraph_mcm_frontend/passes/convert_to_mcm_fc.hpp"
#include "ngraph_mcm_frontend/ops/mcm_fc.hpp"
#include "ngraph_mcm_frontend/ops/mcm_bias.hpp"

#include <legacy/ngraph_ops/fully_connected.hpp>
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

// clang-format on
