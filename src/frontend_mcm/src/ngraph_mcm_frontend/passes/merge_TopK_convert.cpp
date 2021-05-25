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
