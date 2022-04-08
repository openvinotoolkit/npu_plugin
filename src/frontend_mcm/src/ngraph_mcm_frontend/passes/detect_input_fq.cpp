//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "ngraph_mcm_frontend/passes/detect_input_fq.hpp"
#include <ngraph/op/constant.hpp>
#include <ngraph/op/fake_quantize.hpp>
#include <vector>

namespace {
bool is_0_255_range(const std::vector<double>& input_low_values, const std::vector<double>& input_high_values) {
    const auto acceptableLowValues =
            std::none_of(input_low_values.cbegin(), input_low_values.cend(), [](const double value) -> bool {
                return value > 5. || value < 0.;
            });
    const auto acceptableHighValues =
            std::none_of(input_high_values.cbegin(), input_high_values.cend(), [](const double value) -> bool {
                return value < 250. || value > 255.;
            });

    return acceptableLowValues && acceptableHighValues;
}
}  // namespace

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

            if (!is_0_255_range(input_low_values, input_high_values)) {
                return false;
            }

            *_needConvertInputPrecision = true;
            break;
        }
    }
    return true;
}
