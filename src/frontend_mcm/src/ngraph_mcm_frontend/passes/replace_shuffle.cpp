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

// clang-format off

#include <details/ie_exception.hpp>
#include "ngraph_mcm_frontend/passes/replace_shuffle.hpp"

#include <memory>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/reshape.hpp>
#include <ngraph/op/transpose.hpp>

bool ReplaceShuffle::run_on_node(std::shared_ptr<ngraph::Node> node) {
    // replace 5-d { Reshape -> Transpose } sub-graphs with 4-d equivalents
    // Reshape { NxCxHxW -> NxCxDxHxW } becomes { NxCxHxW -> NxCxDxH*W }
    // Transpose { NxCxDxHxW -> 0, 2, 1, 3, 4 } becomes { NxCxDxH*W -> 0, 2, 1, 3 }
    // this is required to drop 5-d permutations and reshapes
    // FIXME: support 5-d operations
    if (std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(node) != nullptr) {
        auto transpose_inputs = node->input_values();
        if (transpose_inputs.size() != 2) {
            return false;
        }
        auto reshape_node = std::dynamic_pointer_cast<ngraph::op::v1::Reshape>(transpose_inputs.at(0).get_node_shared_ptr());
        if (reshape_node == nullptr) {
            return false;
        }
        auto reshape_shape = reshape_node->get_shape();
        if (reshape_shape.size() != 5) {
            return false;
        }
        auto transpose_shape = node->get_shape();
        if (transpose_shape.size() != 5) {
            return false;
        }
        auto transpose_order = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(transpose_inputs.at(1).get_node_shared_ptr());
        if (transpose_order == nullptr) {
            return false;
        }
        auto transpose_order_val = transpose_order->cast_vector<int64_t>();
        // this check means that dimensions H and W are not going to be permuted
        if (transpose_order_val[3] != 3 || transpose_order_val[4] != 4) {
            return false;
        }

        auto reshape_inputs = reshape_node->input_values();
        auto new_reshape_shape_val = std::vector<uint64_t>{
            reshape_shape.at(0),
            reshape_shape.at(1),
            reshape_shape.at(2),
            reshape_shape.at(3) * reshape_shape.at(4),
        };
        auto new_reshape_shape = std::make_shared<ngraph::op::v0::Constant>(
            ngraph::element::u64,
            ngraph::Shape{new_reshape_shape_val.size()},
            new_reshape_shape_val.data());
        auto new_reshape = std::make_shared<ngraph::op::v1::Reshape>(reshape_inputs.at(0).get_node_shared_ptr()->output(0),
            new_reshape_shape, false);

        // omit last order position to convert { 0 2 1 3 4 } to { 0 2 1 3 }
        auto new_transpose_order_val = std::vector<int64_t>(transpose_order_val.cbegin(), transpose_order_val.cend() - 1);
        auto new_transpose_order = std::make_shared<ngraph::op::v0::Constant>(
            ngraph::element::i64,
            ngraph::Shape{new_transpose_order_val.size()},
            new_transpose_order_val.data());

        auto new_transpose = std::make_shared<ngraph::op::v1::Transpose>(new_reshape->output(0), new_transpose_order);
        ngraph::replace_node(reshape_node, new_reshape);
        ngraph::replace_node(node, new_transpose);

        return true;
    }
    return false;
}

// clang-format on
