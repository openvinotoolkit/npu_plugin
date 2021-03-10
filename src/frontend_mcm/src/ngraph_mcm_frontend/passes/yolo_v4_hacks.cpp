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

#include "ngraph_mcm_frontend/passes/yolo_v4_hacks.hpp"

#include <algorithm>
#include <memory>

#include "ngraph/opsets/opset1.hpp"
#include "ngraph/pattern/op/true.hpp"
#include "ngraph/pattern/op/wrap_type.hpp"
#include "ngraph/rt_info.hpp"

static bool process_interp(const std::shared_ptr<ngraph::op::v0::Interpolate>& interp) {
    const auto input_values = interp->input_values();
    const auto strides = ngraph::Strides{1, 1};
    const auto pads_begin = ngraph::Shape{0, 0};
    const auto pads_end = ngraph::Shape{0, 0};
    const auto kernel = ngraph::Shape{1, 1};
    const auto rounding_mode = ngraph::op::RoundingType::FLOOR;
    const auto auto_pad = ngraph::op::PadType::EXPLICIT;
    const auto pre_max_pool_1 =
            std::make_shared<ngraph::op::v1::MaxPool>(input_values.at(0).get_node_shared_ptr()->output(0), strides,
                                                      pads_begin, pads_end, kernel, rounding_mode, auto_pad);

    const auto pre_max_pool_2 = std::make_shared<ngraph::op::v1::MaxPool>(
            pre_max_pool_1->output(0), strides, pads_begin, pads_end, kernel, rounding_mode, auto_pad);

    ngraph::OutputVector new_inputs = {};
    new_inputs.push_back(pre_max_pool_2->output(0));
    for (size_t i = 1; i < input_values.size(); i++) {
        new_inputs.push_back(input_values.at(i));
    }

    const auto new_interp = interp->clone_with_new_inputs(new_inputs);
    const auto post_max_pool_1 = std::make_shared<ngraph::op::v1::MaxPool>(new_interp->output(0), strides, pads_begin,
                                                                           pads_end, kernel, rounding_mode, auto_pad);

    const auto post_max_pool_2 = std::make_shared<ngraph::op::v1::MaxPool>(
            post_max_pool_1->output(0), strides, pads_begin, pads_end, kernel, rounding_mode, auto_pad);

    ngraph::replace_node(interp, post_max_pool_2);
    return true;
}

ngraph::pass::YoloV4Hacks::YoloV4Hacks() {
    ngraph::graph_rewrite_callback replace_cb = [](pattern::Matcher& m) {
        auto match_root = m.get_match_root();
        const auto interp = std::dynamic_pointer_cast<ngraph::op::v0::Interpolate>(match_root);
        if (interp) {
            return process_interp(interp);
        }

        return false;
    };

    auto all_nodes = std::make_shared<ngraph::pattern::op::True>();
    auto match = std::make_shared<ngraph::pattern::Matcher>(all_nodes, "mcmAdaptation.YoloV4Hacks");
    this->register_matcher(match, replace_cb);
}
