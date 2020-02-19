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

#include "ngraph_mcm_frontend/passes/quantize_constants.hpp"
#include "ngraph_mcm_frontend/ops/mcm_quantize.hpp"
#include "ngraph_mcm_frontend/quantization_helpers.hpp"
#include "ngraph_mcm_frontend/mcm_attrs.hpp"
#include "ngraph_mcm_frontend/mcm_helpers.hpp"
#include <details/ie_exception.hpp>
#include <memory>
#include <vector>

namespace {

bool rewrite(ngraph::pattern::Matcher& m) {
    const auto quantize = std::dynamic_pointer_cast<McmQuantize>(m.get_match_root());
    IE_ASSERT(quantize != nullptr);

    const auto constant = std::dynamic_pointer_cast<ngraph::op::Constant>(quantize->input_value(0).get_node_shared_ptr());
    const auto scales = std::dynamic_pointer_cast<ngraph::op::Constant>(quantize->input_value(1).get_node_shared_ptr());
    const auto zeroPoints = std::dynamic_pointer_cast<ngraph::op::Constant>(quantize->input_value(2).get_node_shared_ptr());
    IE_ASSERT(constant != nullptr || scales != nullptr || zeroPoints != nullptr);
    IE_ASSERT(scales->get_shape() == zeroPoints->get_shape());

    const auto constantData = constant->cast_vector<double>();
    const auto scalesData = scales->cast_vector<double>();
    const auto zeroPointsData = zeroPoints->cast_vector<int64_t>();

    const auto quantizedData = quantizeData(
        quantize->get_shape(), quantize->get_element_type(),
        constantData, constant->get_shape(),
        scalesData, zeroPointsData,
        scales->get_shape());

    const auto quantizedConstant = std::make_shared<ngraph::op::Constant>(
        quantize->get_element_type(), quantize->get_shape(), quantizedData);
    quantizedConstant->set_friendly_name(constant->get_friendly_name());

    const auto quantParams = makeQuantParams(zeroPointsData, scalesData);
    McmOpAttrs::setQuantParams(quantParams, quantizedConstant);

    ngraph::replace_node(quantize, quantizedConstant);

    return true;
}

}  // namespace

QuantizeConstants::QuantizeConstants() {
    const std::vector<double> fakeData(1);

    const auto data = std::make_shared<ngraph::op::Constant>(
        ngraph::element::f32, ngraph::Shape {1}, fakeData.data());

    const auto scales = std::make_shared<ngraph::op::Constant>(
        ngraph::element::f32, ngraph::Shape {1}, fakeData.data());

    const auto zeroPoints = std::make_shared<ngraph::op::Constant>(
        ngraph::element::u8, ngraph::Shape {1}, fakeData.data());

    const auto quantize = std::make_shared<McmQuantize>(
        data, scales, zeroPoints, ngraph::element::u8);

    const auto m = std::make_shared<ngraph::pattern::Matcher>(quantize);
    add_matcher(m, rewrite, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
}

#endif
// clang-format on
