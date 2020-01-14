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

#include "ngraph_mcm_frontend/passes/split_fq.hpp"
#include "ngraph_mcm_frontend/ops/mcm_quantize.hpp"
#include "ngraph_mcm_frontend/ops/mcm_dequantize.hpp"
#include "ngraph_mcm_frontend/quantization_helpers.hpp"
#include <details/ie_exception.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/fused/fake_quantize.hpp>
#include <memory>

bool SplitFQ::run_on_node(std::shared_ptr<ngraph::Node> node) {
    if (const auto fq = std::dynamic_pointer_cast<ngraph::op::FakeQuantize>(node)) {
        const auto data = fq->input_value(0).get_node_shared_ptr();
        IE_ASSERT(data != nullptr);

        const auto inputLow = std::dynamic_pointer_cast<ngraph::op::Constant>(fq->input_value(1).get_node_shared_ptr());
        const auto inputHigh = std::dynamic_pointer_cast<ngraph::op::Constant>(fq->input_value(2).get_node_shared_ptr());
        const auto outputLow = std::dynamic_pointer_cast<ngraph::op::Constant>(fq->input_value(3).get_node_shared_ptr());
        const auto outputHigh = std::dynamic_pointer_cast<ngraph::op::Constant>(fq->input_value(4).get_node_shared_ptr());
        IE_ASSERT(inputLow != nullptr && inputHigh != nullptr && outputLow != nullptr && outputHigh != nullptr);
        IE_ASSERT(inputLow->get_shape() == inputHigh->get_shape());
        IE_ASSERT(outputLow->get_shape() == outputHigh->get_shape());

        const auto levels = fq->get_levels();

        // TODO: other types?
        const auto elemType = ngraph::element::u8;

        const auto inputLowData = inputLow->cast_vector<double>();
        const auto inputHighData = inputHigh->cast_vector<double>();
        const auto outputLowData = outputLow->cast_vector<double>();
        const auto outputHighData = outputHigh->cast_vector<double>();

        const auto inputScalesData = calcScales(inputLowData, inputHighData, levels);
        const auto outputScalesData = calcScales(outputLowData, outputHighData, levels);

        const auto inputScales = std::make_shared<ngraph::op::Constant>(ngraph::element::f64, inputLow->get_shape(), inputScalesData);
        const auto outputScales = std::make_shared<ngraph::op::Constant>(ngraph::element::f64, outputLow->get_shape(), outputScalesData);

        const auto inputZeroPointsData = calcZeroPoints(inputLowData, inputHighData, levels, elemType);
        const auto outputZeroPointsData = calcZeroPoints(outputLowData, outputHighData, levels, elemType);

        const auto inputZeroPoints = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, inputLow->get_shape(), inputZeroPointsData);
        const auto outputZeroPoints = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, outputLow->get_shape(), outputZeroPointsData);

        const auto quantize = std::make_shared<McmQuantize>(
            data, inputScales, inputZeroPoints, elemType);
        quantize->set_friendly_name(fq->get_friendly_name() + "_quantize");

        const auto dequantize = std::make_shared<McmDequantize>(
            quantize, outputScales, outputZeroPoints, data->get_element_type());
        dequantize->set_friendly_name(fq->get_friendly_name() + "_dequantize");

        ngraph::replace_node(fq, dequantize);

        return true;
    }

    return false;
}

#endif
// clang-format on
