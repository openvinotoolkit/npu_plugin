//
// Copyright 2021 Intel Corporation.
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

// clang-format on

#include "ngraph_mcm_frontend/passes/replace_shuffle.hpp"

#include <memory>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/reshape.hpp>
#include <ngraph/op/transpose.hpp>

namespace {
bool isShufflingChannels(const std::vector<int64_t>& permutation) {
    const std::vector<int64_t> shuffle_channels_permute = {0, 2, 1, 3, 4};
    return std::equal(permutation.begin(), permutation.end(), shuffle_channels_permute.begin());
}
}  // namespace

bool ReplaceShuffle::run_on_node(std::shared_ptr<ngraph::Node> node) {
    // replace 5-d { Reshape -> Transpose } sub-graphs with 4-d equivalents
    // Reshape { NxCxHxW -> NxCxDxHxW } becomes { NxCxHxW -> NxCxDxH*W }
    // Transpose { NxCxDxHxW -> 0, 2, 1, 3, 4 } becomes { NxCxDxH*W -> 0, 2, 1, 3 }
    // this is required to drop 5-d permutations and reshapes
    // FIXME: support 5-d operations
    if (std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(node) != nullptr) {
        const auto transpose_inputs = node->input_values();
        if (transpose_inputs.size() != 2) {
            return false;
        }
        const auto reshape_node =
                std::dynamic_pointer_cast<ngraph::op::v1::Reshape>(transpose_inputs.at(0).get_node_shared_ptr());
        const auto transpose_order =
                std::dynamic_pointer_cast<ngraph::op::v0::Constant>(transpose_inputs.at(1).get_node_shared_ptr());
        if (reshape_node == nullptr || transpose_order == nullptr) {
            return false;
        }

        const auto reshape_dims = reshape_node->get_shape();
        const auto transpose_dims = node->get_shape();
        if (reshape_dims.size() != 5 || transpose_dims.size() != 5) {
            return false;
        }

        const auto transpose_order_val = transpose_order->cast_vector<int64_t>();
        // check that we're actually going to shuffle channels
        if (!isShufflingChannels(transpose_order_val)) {
            return false;
        }

        const auto reshape_inputs = reshape_node->input_values();
        const auto new_reshape_dims_val = std::vector<uint64_t>{
                reshape_dims.at(0),
                reshape_dims.at(1),
                reshape_dims.at(2),
                reshape_dims.at(3) * reshape_dims.at(4),
        };
        const auto new_reshape_dims = std::make_shared<ngraph::op::v0::Constant>(
                ngraph::element::u64, ngraph::Shape{new_reshape_dims_val.size()}, new_reshape_dims_val.data());
        const auto new_reshape = std::make_shared<ngraph::op::v1::Reshape>(
                reshape_inputs.at(0).get_node_shared_ptr()->output(0), new_reshape_dims, false);

        // omit last order position to convert { 0 2 1 3 4 } to { 0 2 1 3 }
        const auto new_transpose_order_val =
                std::vector<int64_t>(transpose_order_val.cbegin(), transpose_order_val.cend() - 1);
        const auto new_transpose_order = std::make_shared<ngraph::op::v0::Constant>(
                ngraph::element::i64, ngraph::Shape{new_transpose_order_val.size()}, new_transpose_order_val.data());

        const auto new_transpose =
                std::make_shared<ngraph::op::v1::Transpose>(new_reshape->output(0), new_transpose_order);
        ngraph::replace_node(reshape_node, new_reshape);
        ngraph::replace_node(node, new_transpose);

        return true;
    }
    return false;
}
