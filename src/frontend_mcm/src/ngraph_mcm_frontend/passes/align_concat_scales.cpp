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

#include <ie_common.h>
#include "ngraph_mcm_frontend/passes/align_concat_scales.hpp"

#include <memory>
#include <ngraph/op/fake_quantize.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/type/element_type.hpp>
#include "ngraph/op/concat.hpp"
#include <legacy/ngraph_ops/prior_box_ie.hpp>

#include "ngraph_mcm_frontend/quantization_helpers.hpp"

#include <ngraph/op/prior_box.hpp>

namespace {

bool needsConcatScaleAlignment(const std::shared_ptr<ngraph::Node>& node) {
    auto input_values = node->input_values();
    for ( auto&& iv : input_values ) {
        if (dynamic_cast<ngraph::op::v0::PriorBox*>(iv.get_node()) || dynamic_cast<ngraph::op::PriorBoxIE*>(iv.get_node()))
            return false;
    }
    return true;
}

bool inputsHasSameScalesAndZeroPoints(const std::vector<std::shared_ptr<ngraph::Node>>& inputs) {
        if (inputs.size() < 2) return true;

        auto fq1 = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(inputs[0]);
        if (fq1 == nullptr) {
            IE_THROW() << "Failed to cast first input to FakeQuantize";
        }
        auto outputLow1 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq1->input_value(3).get_node_shared_ptr());
        auto outputHigh1 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq1->input_value(4).get_node_shared_ptr());
        auto ol = outputLow1->cast_vector<double>().at(0);
        auto oh = outputHigh1->cast_vector<double>().at(0);

        for (size_t i = 1; i < inputs.size(); i++) {
            auto fq2 = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(inputs[i]);
            if (fq2 == nullptr) {
                IE_THROW() << "Failed to cast input to FakeQuantize";
            }

            auto outputLow2 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq2->input_value(3).get_node_shared_ptr());
            auto outputHigh2 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq2->input_value(4).get_node_shared_ptr());
            auto outputLowValues2 = outputLow2->cast_vector<double>();
            auto outputHighValues2 = outputHigh2->cast_vector<double>();

            for (size_t c = 0; c < outputLowValues2.size(); c++) {
                if ((outputLowValues2[c] != ol) || (outputHighValues2[c] != oh)) {
                    return false;
                }
            }
            if (fq1->get_levels() != fq2->get_levels()) {
                return false;
            }
        }
    return true;
}

void setFakeQuantizeParams(
        const std::shared_ptr<ngraph::op::v0::FakeQuantize>& fq,
        const size_t& maxLevels,
        const double& minVal,
        const double& maxVal) {
    auto inputLow = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq->input_value(1).get_node_shared_ptr());
    auto inputHigh = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq->input_value(2).get_node_shared_ptr());
    auto outputLow = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq->input_value(3).get_node_shared_ptr());
    auto outputHigh = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq->input_value(4).get_node_shared_ptr());

    std::vector<float> scaledInputLowValues(ngraph::shape_size(inputLow->get_output_shape(0)), minVal);
    std::vector<float> scaledInputHighValues(ngraph::shape_size(inputHigh->get_output_shape(0)), maxVal);
    std::vector<float> scaledOutputLowValues(ngraph::shape_size(outputLow->get_output_shape(0)), minVal);
    std::vector<float> scaledOutputHighValues(ngraph::shape_size(outputHigh->get_output_shape(0)), maxVal);

    auto newInputLow = std::make_shared<ngraph::op::v0::Constant>(
        ngraph::element::f32,
        inputLow->get_output_partial_shape(0).to_shape(),
        scaledInputLowValues.data());
    auto newInputHigh = std::make_shared<ngraph::op::v0::Constant>(
        ngraph::element::f32,
        inputHigh->get_output_partial_shape(0).to_shape(),
        scaledInputHighValues.data());
    auto newOutputLow = std::make_shared<ngraph::op::v0::Constant>(
        ngraph::element::f32,
        outputLow->get_output_partial_shape(0).to_shape(),
        scaledOutputLowValues.data());
    auto newOutputHigh = std::make_shared<ngraph::op::v0::Constant>(
        ngraph::element::f32,
        outputHigh->get_output_partial_shape(0).to_shape(),
        scaledOutputHighValues.data());

    newInputLow->set_friendly_name(inputLow->get_friendly_name() + "_aligned");
    newInputHigh->set_friendly_name(inputHigh->get_friendly_name() + "_aligned");
    newOutputLow->set_friendly_name(outputLow->get_friendly_name() + "_aligned");
    newOutputHigh->set_friendly_name(outputHigh->get_friendly_name() + "_aligned");

    fq->set_levels(maxLevels);
    ngraph::replace_node(inputLow, newInputLow);
    ngraph::replace_node(inputHigh, newInputHigh);
    ngraph::replace_node(outputLow, newOutputLow);
    ngraph::replace_node(outputHigh, newOutputHigh);
    }
}

bool AlignConcatScales::run_on_node(std::shared_ptr<ngraph::Node> node)
{
    if (std::dynamic_pointer_cast<ngraph::op::v0::Concat>(node) != nullptr) {
        if (!needsConcatScaleAlignment(node)) {
            return false;
        }

        auto inputs = getInputsFQ(node);

        if (inputsHasSameScalesAndZeroPoints(inputs)) {
            return false;
        }

        size_t maxLevels = 0;
        double minVal = std::numeric_limits<double>::max();
        double maxVal = std::numeric_limits<double>::min();
        for (const auto& input : inputs) {
            auto fq = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(input);
            auto outputLow = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq->input_value(3).get_node_shared_ptr());
            auto outputHigh = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq->input_value(4).get_node_shared_ptr());
            auto outputLowValues = outputLow->cast_vector<double>();
            auto outputHighValues = outputHigh->cast_vector<double>();
            auto levels = fq->get_levels();
            if (maxLevels < levels) maxLevels = levels;

            for (size_t i = 0; i < outputLowValues.size(); i++) {
                double ol = outputLowValues[i];
                double oh = outputHighValues[i];
                if (minVal > ol) minVal = ol;
                if (maxVal < oh) maxVal = oh;
            }
        }
        for (const auto& input : inputs) {
            auto fq = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(input);
            setFakeQuantizeParams(fq, maxLevels, minVal, maxVal);
        }
        return true;
    }
    return false;
}

// clang-format on
