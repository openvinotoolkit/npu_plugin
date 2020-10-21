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
        auto last_reshape = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(pattern_to_output.at(reshape4).get_node_shared_ptr());

        std::vector<size_t> first_reshape_in_dims = first_reshape->get_input_shape(0);
        std::vector<size_t> last_reshape_out_dims = last_reshape->get_output_shape(0);

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
