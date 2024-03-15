//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/passes/convert_variadic_split_to_strided_slice.hpp"

#include <memory>
#include <openvino/op/constant.hpp>
#include <openvino/op/strided_slice.hpp>
#include <openvino/op/variadic_split.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include "openvino/core/node.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {

namespace passes {

void replaceWithStridedSlice(int64_t axis, std::shared_ptr<ov::Node> variadicSplit) {
    auto shapeDim = variadicSplit->input_value(0).get_shape().size();
    int64_t dim = static_cast<int64_t>(shapeDim);
    VPUX_THROW_UNLESS((axis >= -dim) && (axis <= (dim - 1)),
                      "VariadicSplit node '{0}' has unsupported axis value '{1}'", variadicSplit->get_friendly_name(),
                      axis);
    if (axis < 0)
        axis = shapeDim + axis;

    std::vector<size_t> startCoords(shapeDim);

    std::vector<int64_t> begin_mask(shapeDim, 1);
    std::vector<int64_t> end_mask(shapeDim, 1);
    std::vector<int64_t> new_axis_mask = {};
    std::vector<int64_t> shrink_axis_mask = {};
    std::vector<int64_t> ellipsis_mask = {};
    begin_mask[axis] = 0;
    end_mask[axis] = 0;

    for (size_t i = 0; i < variadicSplit->get_output_size(); ++i) {
        ov::Shape beginShape(startCoords);
        const auto beginOp = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{shapeDim}, beginShape);
        ov::Shape endShape(variadicSplit->get_output_shape(i));
        endShape[axis] += beginShape[axis];
        const auto endOp = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{shapeDim}, endShape);

        auto stridedSlice = std::make_shared<ov::op::v1::StridedSlice>(
                variadicSplit->input_value(0), beginOp->output(0), endOp->output(0), begin_mask, end_mask,
                new_axis_mask, shrink_axis_mask, ellipsis_mask);
        stridedSlice->set_friendly_name(variadicSplit->get_friendly_name() + "." + std::to_string(i));
        for (auto& input : variadicSplit->output(i).get_target_inputs()) {
            input.replace_source_output(stridedSlice->output(0));
        }
        startCoords[axis] += variadicSplit->get_output_shape(i)[axis];
    }
}

ConvertVariadicSplitToStridedSliceOp::ConvertVariadicSplitToStridedSliceOp() {
    auto variadicSplit = ov::pass::pattern::wrap_type<ov::op::v1::VariadicSplit>();

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto variadicSplit = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(m.get_match_root());
        if (variadicSplit == nullptr) {
            return false;
        }

        VPUX_THROW_UNLESS(variadicSplit->get_input_size() == 3,
                          "VariadicSplit node '{0}' has unsupported number of inputs'{1}'",
                          variadicSplit->get_friendly_name(), variadicSplit->get_input_size());

        // Find axis.
        const auto axis_node = variadicSplit->input_value(1).get_node_shared_ptr();
        const auto axis_node_const = ov::as_type_ptr<ov::op::v0::Constant>(axis_node);

        int64_t axis = 0;
        if (axis_node->get_element_type() == ov::element::i32) {
            axis = static_cast<int64_t>(axis_node_const->get_data_ptr<int32_t>()[0]);
        } else if (axis_node->get_element_type() == ov::element::i64) {
            axis = axis_node_const->get_data_ptr<int64_t>()[0];
        }
        replaceWithStridedSlice(axis, variadicSplit);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(variadicSplit, "ConvertVariadicSplitToStridedSliceOp");
    register_matcher(m, callback);
}

}  // namespace passes
}  // namespace vpux
