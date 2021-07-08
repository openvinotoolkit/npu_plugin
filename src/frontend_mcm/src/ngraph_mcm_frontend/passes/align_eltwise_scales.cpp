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
#include "ngraph_mcm_frontend/passes/align_eltwise_scales.hpp"

#include <memory>
#include <ngraph/ops.hpp>
#include <ngraph/type/element_type.hpp>
#include "ngraph_mcm_frontend/ops/mcm_eltwise.hpp"

#include <legacy/ngraph_ops/eltwise.hpp>

#include "vpux/quantization_helpers.hpp"

namespace {

bool inputsHasSameScales(
        const std::vector<std::shared_ptr<ngraph::Node>>& inputs,
        const size_t& maxValues,
        const size_t& maxValuesIdx) {
        for (size_t i = 0; i < inputs.size(); i++) {
            auto fq1 = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(inputs[i]);
            if (fq1 == nullptr) {
                IE_THROW() << "Failed to cast input to FakeQuantize";
            }
            auto fq2 = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(inputs[maxValuesIdx]);
            if (fq2 == nullptr) {
                IE_THROW() << "Failed to cast input at maxValuesIdx to FakeQuantize";
            }

            auto outputLow1 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq1->input_value(3).get_node_shared_ptr());
            auto outputHigh1 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq1->input_value(4).get_node_shared_ptr());
            auto outputLowValues1 = outputLow1->cast_vector<double>();
            auto outputHighValues1 = outputHigh1->cast_vector<double>();

            auto outputLow2 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq2->input_value(3).get_node_shared_ptr());
            auto outputHigh2 = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq2->input_value(4).get_node_shared_ptr());
            auto outputLowValues2 = outputLow2->cast_vector<double>();
            auto outputHighValues2 = outputHigh2->cast_vector<double>();

            for (size_t c = 0; c < maxValues; c++) {
                size_t c1 = outputHighValues1.size() == 1 ? 0 : c;
                size_t c2 = c;
                if ((outputHighValues1[c1] - outputLowValues1[c1]) !=
                    (outputHighValues2[c2] - outputLowValues2[c2])) {
                    return false;
                }
            }
            if (fq1->get_levels() != fq2->get_levels()) {
                return false;
            }
    }
    return true;
}

void setFakeQuantizeScales(
        const std::shared_ptr<ngraph::op::v0::FakeQuantize>& fq,
        const size_t& maxLevels,
        const std::vector<double>& maxRange) {
    auto inputLow = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq->input_value(1).get_node_shared_ptr());
    auto inputHigh = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq->input_value(2).get_node_shared_ptr());
    auto outputLow = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq->input_value(3).get_node_shared_ptr());
    auto outputHigh = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq->input_value(4).get_node_shared_ptr());
    auto inputLowValues = inputLow->cast_vector<double>();
    auto inputHighValues = inputHigh->cast_vector<double>();
    auto outputLowValues = outputLow->cast_vector<double>();
    auto outputHighValues = outputHigh->cast_vector<double>();
    std::vector<float> scaledInputLowValues(inputLowValues.size());
    std::vector<float> scaledInputHighValues(inputHighValues.size());
    std::vector<float> scaledOutputLowValues(outputLowValues.size());
    std::vector<float> scaledOutputHighValues(outputHighValues.size());

    for (size_t i = 0; i < inputLowValues.size(); i++) {
        double range = inputHighValues[i] - inputLowValues[i];
        double updatedInputLow = inputLowValues[i] * maxRange[i] / range;
        scaledInputLowValues[i] = static_cast<float>(updatedInputLow);
        scaledInputHighValues[i] = static_cast<float>(updatedInputLow + maxRange[i]);
    }

    for (size_t i = 0; i < outputLowValues.size(); i++) {
        double range = outputHighValues[i] - outputLowValues[i];
        double updatedOutputLow = outputLowValues[i] * maxRange[i] / range;
        scaledOutputLowValues[i] = static_cast<float>(updatedOutputLow);
        scaledOutputHighValues[i] = static_cast<float>(updatedOutputLow + maxRange[i]);
    }

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

bool AlignEltwiseScales::run_on_node(std::shared_ptr<ngraph::Node> node)
{
    if (std::dynamic_pointer_cast<McmEltwise>(node) ||
        std::dynamic_pointer_cast<ngraph::op::v1::Add>(node) ||
        std::dynamic_pointer_cast<ngraph::op::Eltwise>(node)) {
        if (std::dynamic_pointer_cast<ngraph::op::Eltwise>(node)->eltwise_type == ELTWISE_TYPE::Prod) {
            return false;
        }

        auto inputs = getInputsFQ(node);

        size_t maxValues = 1;
        size_t maxValuesIdx = 0;
        for (size_t i = 0; i < inputs.size(); i++) {
            auto fq = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(inputs[i]);
            auto lw = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq->input_value(3).get_node_shared_ptr());
            auto outputLowValues = lw->cast_vector<int64_t>().size();

            if (maxValues < outputLowValues) {
                maxValues = outputLowValues;
                maxValuesIdx = i;
            }
        }

        if (inputsHasSameScales(inputs, maxValues, maxValuesIdx)) {
            return false;
        }

        size_t maxLevels = 0;
        std::vector<double> maxRange(maxValues, 0.0);
        for (const auto& input : inputs) {
            auto fq = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(input);
            const unsigned levels = fq->get_levels();
            auto lw = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq->input_value(3).get_node_shared_ptr());
            auto hw = std::dynamic_pointer_cast<ngraph::op::v0::Constant>(fq->input_value(4).get_node_shared_ptr());
            // TBD
            // IE_ASSERT(lw->get_element_type().is_real());
            // IE_ASSERT(hw->get_element_type().is_real());
            auto outputLowValues = lw->cast_vector<double>();
            auto outputHighValues = hw->cast_vector<double>();

            if (maxLevels < levels)
                maxLevels = levels;

            for (size_t i = 0; i < maxValues; i++) {
                size_t c = outputHighValues.size() == 1 ? 0 : i;
                double range = outputHighValues[c] - outputLowValues[c];
                if (maxRange[i] < range)
                    maxRange[i] = range;
            }
        }

        for (const auto& input : inputs) {
            auto fq = std::dynamic_pointer_cast<ngraph::op::v0::FakeQuantize>(input);
            setFakeQuantizeScales(fq, maxLevels, maxRange);
        }

        return true;
    }

    return false;
}

// clang-format on
