//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
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
