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

#include "ngraph_mcm_frontend/passes/merge_TopK_convert.hpp"
#include <details/ie_exception.hpp>
#include <ngraph/op/convert.hpp>
#include <ngraph/op/result.hpp>
#include <ngraph_ops/topk_ie.hpp>
#include <memory>

bool MergeTopKConvert::run_on_function(std::shared_ptr<ngraph::Function> f) {
    for (auto & node : f->get_ordered_ops()) {
        if (node->get_type_info() == ngraph::op::Convert::type_info) {
            const auto topK = std::dynamic_pointer_cast<ngraph::op::TopKIE>(node->input_value(0).get_node_shared_ptr());
            if (topK == nullptr) {
                return false;
            }
            bool success = replace_output_update_name(node->output(0), node->input_value(0));
            return success;
        }
    }
    return false;
}

// clang-format on
