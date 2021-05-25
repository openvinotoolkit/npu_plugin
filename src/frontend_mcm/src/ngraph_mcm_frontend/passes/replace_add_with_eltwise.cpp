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

#include "ngraph_mcm_frontend/passes/replace_add_with_eltwise.hpp"

#include <memory>
#include "ngraph/op/add.hpp"
#include "ngraph_mcm_frontend/ops/mcm_eltwise.hpp"


bool ReplaceAddWithMcmEltwise::run_on_node(std::shared_ptr<ngraph::Node> node)
{
    if (const auto add = std::dynamic_pointer_cast<ngraph::op::v1::Add>(node))
    {
        const auto mcmEltwise = std::make_shared<McmEltwise>(
            add->input_value(0),
            add->input_value(1),
            McmEltwise::OperationType::SUM,
            add->get_output_element_type(0));

        ngraph::replace_node(add, mcmEltwise);
        return true;
    }
    return false;
}

// clang-format on
