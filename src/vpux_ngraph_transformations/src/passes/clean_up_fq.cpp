//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/passes/clean_up_fq.hpp"
#include <memory>
#include <ngraph/op/fake_quantize.hpp>
#include <ngraph/ops.hpp>
#include "vpux/quantization_helpers.hpp"

namespace vpux {
namespace passes {

static bool calc_node(const std::shared_ptr<ngraph::Node>& node) {
    return (std::dynamic_pointer_cast<ngraph::op::v1::Split>(node) == nullptr &&
            std::dynamic_pointer_cast<ngraph::op::v1::StridedSlice>(node) == nullptr &&
            std::dynamic_pointer_cast<ngraph::op::v0::Tile>(node) == nullptr &&
            std::dynamic_pointer_cast<ngraph::op::v1::VariadicSplit>(node) == nullptr &&
            std::dynamic_pointer_cast<ngraph::op::v0::Concat>(node) == nullptr &&
            std::dynamic_pointer_cast<ngraph::op::v0::ReorgYolo>(node) == nullptr &&
            std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(node) == nullptr &&
            std::dynamic_pointer_cast<ngraph::op::v1::Reshape>(node) == nullptr &&
            std::dynamic_pointer_cast<ngraph::op::v0::Squeeze>(node) == nullptr &&
            std::dynamic_pointer_cast<ngraph::op::v0::Unsqueeze>(node) == nullptr &&
            std::dynamic_pointer_cast<ngraph::op::v3::ScatterNDUpdate>(node) == nullptr &&
            std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(node) == nullptr);
}

bool CleanUpFQ::run_on_node(std::shared_ptr<ngraph::Node> node) {
    const auto fq_node = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(node);
    if (fq_node == nullptr)
        return false;
    std::set<std::shared_ptr<ngraph::Node>> fq_nodes = {fq_node};
    if (!all_fqs_have_same_io_params(fq_nodes)) {
        return false;
    }

    if (calc_node(fq_node->input_value(0).get_node_shared_ptr())) {
        return false;
    }

    for (const auto& node_output : fq_node->outputs()) {
        for (auto consumer : node_output.get_target_inputs()) {
            if (calc_node(consumer.get_node()->shared_from_this())) {
                return false;
            }
        }
    }

    return replace_output_update_name(fq_node->output(0), fq_node->input_value(0));
}

}  // namespace passes
}  // namespace vpux
