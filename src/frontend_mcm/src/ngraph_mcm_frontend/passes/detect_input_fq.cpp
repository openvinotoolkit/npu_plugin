//
// Copyright 2021 Intel Corporation.
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

#include "ngraph_mcm_frontend/passes/detect_input_fq.hpp"
#include <ngraph/op/constant.hpp>
#include <ngraph/op/fake_quantize.hpp>
#include <vector>

bool DetectInputFQ::run_on_function(std::shared_ptr<ngraph::Function> f) {
    const auto ops = f->get_ordered_ops();

    for (const auto& op : ops) {
        if (op->get_type_info() == ngraph::op::v0::FakeQuantize::type_info) {
            const auto fq_node = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(op);
            if (!fq_node)
                continue;

            const auto node = fq_node->input_value(0).get_node_shared_ptr();
            const auto parameter_node = std::dynamic_pointer_cast<ngraph::op::v0::Parameter>(node);
            if (parameter_node == nullptr)
                continue;

            const auto input_fq_node1 =
                    std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq_node->input_value(1).get_node_shared_ptr());
            const auto input_fq_node2 =
                    std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq_node->input_value(2).get_node_shared_ptr());
            if (input_fq_node1 == nullptr || input_fq_node2 == nullptr)
                return false;

            const auto& input_low_values = input_fq_node1->cast_vector<double>();
            const auto& input_high_values = input_fq_node2->cast_vector<double>();

            for (const auto& in_min : input_low_values) {
                if (in_min > 5.f || in_min < 0.f)
                    return false;
            }
            for (const auto& in_max : input_high_values) {
                if (in_max < 250.f || in_max > 255.f)
                    return false;
            }

            *_needConvertInputPrecision = true;
        }
    }
    return true;
}
