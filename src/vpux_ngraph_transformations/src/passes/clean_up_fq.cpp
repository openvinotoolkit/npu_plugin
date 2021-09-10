//
<<<<<<< HEAD
// Copyright Intel Corporation.
=======
// Copyright 2021 Intel Corporation.
>>>>>>> enable propagate FQ pass in MLIR frontend
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

#include "vpux/passes/clean_up_fq.hpp"
#include <ngraph/op/fake_quantize.hpp>
#include <memory>
#include <ngraph/rt_info.hpp>
#include <ngraph/ops.hpp>

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
            std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(node) == nullptr);
}

bool CleanUpFQ::run_on_node(std::shared_ptr<ngraph::Node> node) {
    const auto fq_node = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(node);
    if (fq_node == nullptr)
        return false;

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
