//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// clang-format off

#include "ngraph_mcm_frontend/passes/merge_TopK_convert.hpp"
#include <ngraph/op/convert.hpp>
#include <ngraph/op/result.hpp>
#include <legacy/ngraph_ops/topk_ie.hpp>
#include <ngraph/op/squeeze.hpp>
#include <memory>

bool MergeTopKConvert::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (auto & node : f->get_ordered_ops()) {
        if (node->get_type_info() == ngraph::op::Convert::type_info) {

            auto input_node = node->input_value(0).get_node_shared_ptr();

            const auto squeeze = std::dynamic_pointer_cast<ngraph::op::Squeeze>(input_node);
            if (squeeze) {
                return replace_output_update_name(node->output(0), node->input_value(0));
            }

            const auto topK = std::dynamic_pointer_cast<ngraph::op::TopKIE>(input_node);
            if (!topK) {
                return false;
            }
            return replace_output_update_name(node->output(0), node->input_value(0));
        }
    }
    return false;
}

// clang-format on
