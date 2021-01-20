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

#include "ngraph_mcm_frontend/passes/handle_3d_transpose.hpp"
#include <ie_layouts.h>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/squeeze.hpp>
#include <ngraph/op/transpose.hpp>

// [Squeeze 3D -> Transpose 3D] -> [Transpose 4D -> Squeeze 3D]

bool Handle3DTranspose::run_on_function(std::shared_ptr<ngraph::Function> f) {
    bool status = false;
    for (auto& node : f->get_ordered_ops()) {
        auto transpose = std::dynamic_pointer_cast<ngraph::op::v1::Transpose>(node);

        if (!transpose || transpose->get_input_size() < 2)
            continue;

        // handling here only single image layout

        auto squeeze =
                std::dynamic_pointer_cast<ngraph::op::v0::Squeeze>(transpose->input_value(0).get_node_shared_ptr());
        if (!squeeze || squeeze->get_input_size() < 2 || squeeze->get_shape().size() != 3)
            continue;

        // in mcm 3d squeeze will be substituted with 4d, but  for transpose it's essential to
        // check previous and current permutation with valid dimensions
        auto transpose_order =
                std::dynamic_pointer_cast<ngraph::op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr());
        if (!transpose_order)
            continue;
        auto transpose_order_val = transpose_order->cast_vector<int64_t>();

        std::vector<int64_t> new_order = {0};
        new_order.reserve(4);
        std::transform(transpose_order_val.begin(), transpose_order_val.end(), std::back_inserter(new_order),
                       [](int64_t val) {
                           return ++val;
                       });

        auto new_transpose_order = std::make_shared<ngraph::op::v0::Constant>(
                ngraph::element::i64, ngraph::Shape{new_order.size()}, new_order.data());

        auto new_transpose = std::make_shared<ngraph::op::v1::Transpose>(
                squeeze->input_value(0).get_node_shared_ptr()->output(0), new_transpose_order);

        status = ngraph::replace_output_update_name(squeeze->output(0), squeeze->input_value(0));

        auto new_squeeze_axis = std::make_shared<ngraph::op::v0::Constant>(ngraph::element::i64, ngraph::Shape{1}, 0);
        auto new_squeeze = std::make_shared<ngraph::op::v0::Squeeze>(new_transpose->output(0), new_squeeze_axis);
        ngraph::replace_node(node, new_squeeze);
    }

    return status;
}
