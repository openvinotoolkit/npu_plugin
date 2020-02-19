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
#ifdef ENABLE_MCM_COMPILER

#include "ngraph_mcm_frontend/passes/merge_result_convert.hpp"
#include "ngraph_mcm_frontend/ops/mcm_conv.hpp"
#include "ngraph_mcm_frontend/ops/mcm_bias.hpp"
#include <details/ie_exception.hpp>
#include <ngraph/op/convert.hpp>
#include <ngraph/op/result.hpp>
#include <memory>

namespace {

bool rewrite(ngraph::pattern::Matcher& m) {
    const auto result = std::dynamic_pointer_cast<ngraph::op::Result>(m.get_match_root());
    IE_ASSERT(result != nullptr);

    const auto convert = std::dynamic_pointer_cast<ngraph::op::Convert>(result->input_value(0).get_node_shared_ptr());
    IE_ASSERT(convert != nullptr);

    // TODO: check actual type.

    const auto op = convert->input_value(0).get_node_shared_ptr();
    IE_ASSERT(op != nullptr);

    if (const auto conv = std::dynamic_pointer_cast<McmConv>(op)) {
        conv->setElemType(convert->get_element_type());
        result->input(0).replace_source_output(conv);
        return true;
    } else if (const auto bias = std::dynamic_pointer_cast<McmBias>(op)) {
        bias->setElemType(convert->get_element_type());
        result->input(0).replace_source_output(bias);
        return true;
    }

    return false;
}

}  // namespace

MergeResultConvert::MergeResultConvert() {
    const auto op = std::make_shared<ngraph::pattern::op::Label>(ngraph::element::f32, ngraph::Shape());
    const auto convert = std::make_shared<ngraph::op::Convert>(op, ngraph::element::f32);
    const auto result = std::make_shared<ngraph::op::Result>(convert);

    const auto m = std::make_shared<ngraph::pattern::Matcher>(result);
    add_matcher(m, rewrite, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}

#endif
// clang-format on
