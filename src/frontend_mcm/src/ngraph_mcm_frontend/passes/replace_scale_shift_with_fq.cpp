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

#include "ngraph_mcm_frontend/passes/replace_scale_shift_with_fq.hpp"
#include <details/ie_exception.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/convert.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/fused/fake_quantize.hpp>
#include <ngraph_ops/scaleshift.hpp>
#include <vector>
#include <numeric>
#include <memory>
#include <limits>

namespace {

bool rewrite(ngraph::pattern::Matcher& m) {
    const auto scaleShift = std::dynamic_pointer_cast<ngraph::op::ScaleShiftIE>(m.get_match_root());
    IE_ASSERT(scaleShift != nullptr);

    const auto convert = std::dynamic_pointer_cast<ngraph::op::v0::Convert>(scaleShift->input_value(0).get_node_shared_ptr());
    const auto scale = std::dynamic_pointer_cast<ngraph::op::Constant>(scaleShift->input_value(1).get_node_shared_ptr());
    const auto shift = std::dynamic_pointer_cast<ngraph::op::Constant>(scaleShift->input_value(2).get_node_shared_ptr());
    IE_ASSERT(convert != nullptr && scale != nullptr && shift != nullptr);

    const auto param = std::dynamic_pointer_cast<ngraph::op::Parameter>(convert->input_value(0).get_node_shared_ptr());
    IE_ASSERT(param != nullptr);

    // TODO: i8?
    if (param->get_element_type() != ngraph::element::u8) {
        return false;
    }

    const auto scaleData = scale->cast_vector<double>();
    const auto shiftData = shift->cast_vector<double>();

    // TODO: is this correct way?
    const auto scaleMeanValue = std::accumulate(scaleData.begin(), scaleData.end(), 0.0) / scaleData.size();
    const auto shiftMeanValue = std::accumulate(shiftData.begin(), shiftData.end(), 0.0) / shiftData.size();

    // TODO: i8?
    const auto minValue = std::numeric_limits<uint8_t>::min() * scaleMeanValue + shiftMeanValue;
    const auto maxValue = std::numeric_limits<uint8_t>::max() * scaleMeanValue + shiftMeanValue;

    const auto low = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape {1}, std::vector<double> {minValue});
    const auto high = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape {1}, std::vector<double> {maxValue});

    // TODO: other values?
    const size_t levels = 256;

    const auto fq = std::make_shared<ngraph::op::FakeQuantize>(convert, low, high, low, high, levels);
    fq->set_friendly_name(scaleShift->get_friendly_name());

    ngraph::replace_node(scaleShift, fq);

    return true;
}

}  // namespace

ReplaceScaleShiftWithFQ::ReplaceScaleShiftWithFQ() {
    const std::vector<double> fakeData(1);

    const auto param = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::u8, ngraph::PartialShape());

    const auto convert = std::make_shared<ngraph::op::v0::Convert>(
        param, ngraph::element::f32);

    const auto scale = std::make_shared<ngraph::op::Constant>(
        ngraph::element::f32, ngraph::Shape {1}, fakeData.data());

    const auto shift = std::make_shared<ngraph::op::Constant>(
        ngraph::element::f32, ngraph::Shape {1}, fakeData.data());

    const auto scaleShift = std::make_shared<ngraph::op::ScaleShiftIE>(
        convert, scale, shift);

    const auto m = std::make_shared<ngraph::pattern::Matcher>(scaleShift);
    add_matcher(m, rewrite, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}

#endif
// clang-format on
