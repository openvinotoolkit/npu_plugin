//
// Copyright 2020 Intel Corporation.
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

#include "ngraph_mcm_frontend/passes/replace_onnx_pattern_to_reorg.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/op/reorg_yolo.hpp>
#include <ngraph/rt_info.hpp>

OnnxReorgPatternToDarkNetReorg::OnnxReorgPatternToDarkNetReorg() {
    auto input = ngraph::pattern::any_input();
    auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(input, ngraph::pattern::wrap_type<ngraph::op::Constant>(), true);
    auto transpose1 = std::make_shared<ngraph::opset1::Transpose>(reshape1, ngraph::pattern::wrap_type<ngraph::op::Constant>());
    auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(transpose1, ngraph::pattern::wrap_type<ngraph::op::Constant>(), true);
    auto transpose2 = std::make_shared<ngraph::opset1::Transpose>(reshape2, ngraph::pattern::wrap_type<ngraph::op::Constant>());
    auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(transpose2, ngraph::pattern::wrap_type<ngraph::op::Constant>(), true);
    auto transpose3 = std::make_shared<ngraph::opset1::Transpose>(reshape3, ngraph::pattern::wrap_type<ngraph::op::Constant>());
    auto reshape4 = std::make_shared<ngraph::opset1::Reshape>(transpose3, ngraph::pattern::wrap_type<ngraph::op::Constant>(), true);
    
    ngraph::graph_rewrite_callback matcher_pass_callback = [=](ngraph::pattern::Matcher& m) {
        auto & pattern_to_output = m.get_pattern_value_map();
        auto reorg_input = pattern_to_output.at(input);

        auto first_reshape = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(pattern_to_output.at(reshape1).get_node_shared_ptr());
        if (first_reshape == nullptr) {
            return false;
        }

        auto last_reshape = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(pattern_to_output.at(reshape4).get_node_shared_ptr());
        if (last_reshape == nullptr) {
            return false;
        }

        std::vector<size_t> first_reshape_in_dims = first_reshape->get_input_shape(0);
        std::vector<size_t> last_reshape_out_dims = last_reshape->get_output_shape(0);

        // Input\Output shapes for Reorg layer from original YoloV2 network
        // Expectations that this pass will only work on the "classic" version of the network
        std::vector<size_t> expected_in_shape = {1, 64, 26, 26};
        std::vector<size_t> expected_out_shape = {1, 256, 13, 13};

        if ((first_reshape_in_dims != expected_in_shape) || (last_reshape_out_dims != expected_out_shape)) {
            return false;
        }

        auto reorgYolo = std::make_shared<ngraph::op::v0::ReorgYolo>(reorg_input, ngraph::Strides{2});

        reorgYolo->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info({pattern_to_output.at(reshape1).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose1).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape2).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose2).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape3).get_node_shared_ptr(),
                                   pattern_to_output.at(transpose3).get_node_shared_ptr(),
                                   pattern_to_output.at(reshape4).get_node_shared_ptr(),
                                  }, reorgYolo);
        ngraph::replace_node(m.get_match_root(), reorgYolo);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshape4, "OnnxReorgPatternToDarkNetReorg");
    register_matcher(m, matcher_pass_callback);
}

// clang-format on
