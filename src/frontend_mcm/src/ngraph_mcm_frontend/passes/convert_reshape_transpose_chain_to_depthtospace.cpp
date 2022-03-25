//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// clang-format on

#include "ngraph_mcm_frontend/passes/convert_reshape_transpose_chain_to_depthtospace.hpp"
#include <ngraph/op/reorg_yolo.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

// it's a workaround for edsr3. check details on ticket EISW-16823
ConvertReshapeTransposeChainToDepthToSpace::ConvertReshapeTransposeChainToDepthToSpace() {
    auto input = ngraph::pattern::any_input();
    auto transpose1 =
            std::make_shared<ngraph::opset1::Transpose>(input, ngraph::pattern::wrap_type<ngraph::op::Constant>());
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(
            transpose1, ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    auto transpose2 =
            std::make_shared<ngraph::opset1::Transpose>(reshape1, ngraph::pattern::wrap_type<ngraph::op::Constant>());
    auto transpose3 =
            std::make_shared<ngraph::opset1::Transpose>(transpose2, ngraph::pattern::wrap_type<ngraph::op::Constant>());
    auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(
            transpose3, ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    auto transpose4 =
            std::make_shared<ngraph::opset1::Transpose>(reshape2, ngraph::pattern::wrap_type<ngraph::op::Constant>());
    auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(
            transpose4, ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    auto transpose5 =
            std::make_shared<ngraph::opset1::Transpose>(reshape3, ngraph::pattern::wrap_type<ngraph::op::Constant>());

    ngraph::graph_rewrite_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto depthtospace_input = pattern_to_output.at(input);

        auto first_transpose = std::dynamic_pointer_cast<ngraph::opset1::Transpose>(
                pattern_to_output.at(transpose1).get_node_shared_ptr());
        if (first_transpose == nullptr) {
            return false;
        }

        auto last_transpose = std::dynamic_pointer_cast<ngraph::opset1::Transpose>(
                pattern_to_output.at(transpose5).get_node_shared_ptr());
        if (last_transpose == nullptr) {
            return false;
        }

        auto depthToSpace = std::make_shared<ngraph::op::v0::DepthToSpace>(depthtospace_input, "DEPTH_FIRST", 2);

        ngraph::copy_runtime_info(
                {
                        pattern_to_output.at(transpose1).get_node_shared_ptr(),
                        pattern_to_output.at(reshape1).get_node_shared_ptr(),
                        pattern_to_output.at(transpose2).get_node_shared_ptr(),
                        pattern_to_output.at(transpose3).get_node_shared_ptr(),
                        pattern_to_output.at(reshape2).get_node_shared_ptr(),
                        pattern_to_output.at(transpose4).get_node_shared_ptr(),
                        pattern_to_output.at(reshape3).get_node_shared_ptr(),
                        pattern_to_output.at(transpose5).get_node_shared_ptr(),
                },
                depthToSpace);
        ngraph::replace_node(m.get_match_root(), depthToSpace);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(transpose5, "ConvertReshapeTransposeChainToDepthToSpace");
    register_matcher(m, matcher_pass_callback);
}
