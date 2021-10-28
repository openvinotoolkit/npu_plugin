//
// Copyright Intel Corporation.
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

#include "ngraph_mcm_frontend/passes/segnet_workaround.hpp"
#include <ngraph/op/reorg_yolo.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

// it's a workaround for segnet.
SegnetWorkaround::SegnetWorkaround() {
    auto input = ngraph::pattern::any_input();
    auto VariadicSplit1 =
            std::make_shared<ngraph::op::v1::VariadicSplit>(input, ngraph::pattern::wrap_type<ngraph::op::Constant>(),
                                                            ngraph::pattern::wrap_type<ngraph::op::Constant>());
    std::cout << "VariadicSplit1" << std::endl;
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(
            VariadicSplit1->output(0), ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(
            reshape1, ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(
            reshape2, ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    auto ScatterNDUpdate1 = std::make_shared<ngraph::op::v3::ScatterNDUpdate>(
            ngraph::pattern::wrap_type<ngraph::op::Constant>(),
            std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape({1, 32, 23, 40, 4})), reshape3);

    // auto reshape4 = std::make_shared<ngraph::opset1::Reshape>(
    //         VariadicSplit1->output(0), ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    // auto reshape5 = std::make_shared<ngraph::opset1::Reshape>(
    //         reshape4, ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    // auto reshape6 = std::make_shared<ngraph::opset1::Reshape>(
    //         reshape5, ngraph::pattern::wrap_type<ngraph::op::Constant>(), false);
    auto ScatterNDUpdate2 = std::make_shared<ngraph::op::v3::ScatterNDUpdate>(
            ScatterNDUpdate1,
            std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape({1, 32, 23, 40, 4})),
            ngraph::pattern::any_input());

    auto ScatterNDUpdate3 = std::make_shared<ngraph::op::v3::ScatterNDUpdate>(
            ScatterNDUpdate2,
            std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape({1, 32, 23, 40, 4})),
            ngraph::pattern::any_input());

    auto ScatterNDUpdate4 = std::make_shared<ngraph::op::v3::ScatterNDUpdate>(
            ScatterNDUpdate3,
            std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape({1, 32, 23, 40, 4})),
            ngraph::pattern::any_input());

    ngraph::graph_rewrite_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto reAllocate_input = pattern_to_output.at(input);

        // auto first_transpose = std::dynamic_pointer_cast<ngraph::opset1::Transpose>(
        //         pattern_to_output.at(transpose1).get_node_shared_ptr());
        // if (first_transpose == nullptr) {
        //     return false;
        // }

        // auto last_transpose = std::dynamic_pointer_cast<ngraph::opset1::Transpose>(
        //         pattern_to_output.at(transpose5).get_node_shared_ptr());
        // if (last_transpose == nullptr) {
        //     return false;
        // }
        std::cout << "find pattern" << std::endl;

        auto reAllocate = std::make_shared<ngraph::op::v0::DepthToSpace>(reAllocate_input, "DEPTH_FIRST", 2);

        // ngraph::copy_runtime_info(
        //         {
        //                 pattern_to_output.at(transpose1).get_node_shared_ptr(),
        //                 pattern_to_output.at(reshape1).get_node_shared_ptr(),
        //                 pattern_to_output.at(transpose2).get_node_shared_ptr(),
        //                 pattern_to_output.at(transpose3).get_node_shared_ptr(),
        //                 pattern_to_output.at(reshape2).get_node_shared_ptr(),
        //                 pattern_to_output.at(transpose4).get_node_shared_ptr(),
        //                 pattern_to_output.at(reshape3).get_node_shared_ptr(),
        //                 pattern_to_output.at(transpose5).get_node_shared_ptr(),
        //         },
        //         depthToSpace);
        ngraph::replace_node(m.get_match_root(), reAllocate);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(ScatterNDUpdate4, "SegnetWorkaround");
    register_matcher(m, matcher_pass_callback);
}
