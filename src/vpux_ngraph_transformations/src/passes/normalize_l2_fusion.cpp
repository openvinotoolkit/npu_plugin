//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux/passes/normalize_l2_fusion.hpp"

#include <details/ie_exception.hpp>
#include <memory>
#include <ngraph/op/constant.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "vpux/utils/core/error.hpp"

namespace vpux {

namespace passes {

NormalizeL2Fusion::NormalizeL2Fusion() {
    auto input = ngraph::pattern::any_input();

    auto axes = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto reduce_l2 = std::make_shared<ngraph::opset4::ReduceL2>(input, axes, true);
    double min, max;
    min = std::numeric_limits<float>::min();
    max = std::numeric_limits<float>::max();
    auto clamp = std::make_shared<ngraph::opset1::Clamp>(reduce_l2, min, max);
    auto divide = std::make_shared<ngraph::opset1::Divide>(input, clamp);

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto data_input = pattern_to_output.at(input);

        auto eps_attr_value = 1e-8;
        ngraph::op::EpsMode mode = ngraph::op::EpsMode::ADD;
        const auto axes_input =
                std::dynamic_pointer_cast<ngraph::opset8::Constant>(pattern_to_output.at(axes).get_node_shared_ptr());

        auto normalize_l2 = std::make_shared<ngraph::opset8::NormalizeL2>(data_input, axes_input, eps_attr_value, mode);
        if (transformation_callback(normalize_l2))
            return false;

        normalize_l2->set_friendly_name(m.get_match_root()->get_friendly_name());
        ngraph::copy_runtime_info(
                {pattern_to_output.at(reduce_l2).get_node_shared_ptr(),
                 pattern_to_output.at(clamp).get_node_shared_ptr(), pattern_to_output.at(divide).get_node_shared_ptr()},
                normalize_l2);
        ngraph::replace_node(m.get_match_root(), normalize_l2);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(divide, "NormalizeL2Fusion");
    register_matcher(m, callback);
}

}  // namespace passes
}  // namespace vpux
