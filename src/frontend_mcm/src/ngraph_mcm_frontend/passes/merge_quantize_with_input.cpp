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

#include "ngraph_mcm_frontend/passes/merge_quantize_with_input.hpp"
#include "ngraph_mcm_frontend/ops/mcm_quantize.hpp"
#include "ngraph_mcm_frontend/mcm_attrs.hpp"
#include "ngraph_mcm_frontend/mcm_helpers.hpp"
#include <details/ie_exception.hpp>
#include <ngraph/op/convert.hpp>
#include <memory>
#include <vector>

namespace {

bool rewrite(ngraph::pattern::Matcher& m) {
    const auto quantize = std::dynamic_pointer_cast<McmQuantize>(m.get_match_root());
    IE_ASSERT(quantize != nullptr);

    const auto convert = std::dynamic_pointer_cast<ngraph::op::v0::Convert>(quantize->input_value(0).get_node_shared_ptr());
    const auto scales = std::dynamic_pointer_cast<ngraph::op::Constant>(quantize->input_value(1).get_node_shared_ptr());
    const auto zeroPoints = std::dynamic_pointer_cast<ngraph::op::Constant>(quantize->input_value(2).get_node_shared_ptr());
    IE_ASSERT(convert != nullptr && scales != nullptr && zeroPoints != nullptr);

    const auto param = std::dynamic_pointer_cast<ngraph::op::Parameter>(convert->input_value(0).get_node_shared_ptr());
    IE_ASSERT(param != nullptr);

    if (param->get_element_type() != quantize->get_element_type()) {
        return false;
    }

    const auto scalesData = scales->cast_vector<double>();
    const auto zeroPointsData = zeroPoints->cast_vector<int64_t>();

    const auto newQuantParams = makeQuantParams(zeroPointsData, scalesData);
    McmOpAttrs::setQuantParams(newQuantParams, param);

    ngraph::replace_node(quantize, param);

    return true;
}

}  // namespace

MergeQuantizeWithInput::MergeQuantizeWithInput() {
    const std::vector<double> fakeData(1);

    const auto param = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::u8, ngraph::PartialShape());

    const auto convert = std::make_shared<ngraph::op::v0::Convert>(
        param, ngraph::element::f32);

    const auto scales = std::make_shared<ngraph::op::Constant>(
        ngraph::element::f32, ngraph::Shape {1}, fakeData.data());

    const auto zeroPoints = std::make_shared<ngraph::op::Constant>(
        ngraph::element::u8, ngraph::Shape {1}, fakeData.data());

    const auto quantize = std::make_shared<McmQuantize>(
        convert, scales, zeroPoints, ngraph::element::u8);

    const auto m = std::make_shared<ngraph::pattern::Matcher>(quantize);
    add_matcher(m, rewrite, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}

#endif
// clang-format on
