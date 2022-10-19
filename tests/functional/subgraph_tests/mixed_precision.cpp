// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>

namespace {

ov::Output<ov::Node> quantize(const ov::Output<ov::Node>& producer, const size_t channels, const bool needQuant) {
    if (!needQuant) {
        // Bypass the quantization
        return producer;
    }

    const ngraph::Shape fqShape = {1, channels, 1, 1};

    const std::vector<ngraph::float16> dataFqInLoVec(channels, 0.f);
    const auto dataFqInLo =
            std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::f16, fqShape, dataFqInLoVec.data());

    const std::vector<ngraph::float16> dataFqInHiVec(channels, 255.f);
    const auto dataFqInHi =
            std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::f16, fqShape, dataFqInHiVec.data());

    const std::vector<ngraph::float16> dataFqOutLoVec(channels, 0.f);
    const auto dataFqOutLo =
            std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::f16, fqShape, dataFqOutLoVec.data());

    const std::vector<ngraph::float16> dataFqOutHiVec(channels, 1.f);
    const auto dataFqOutHi =
            std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::f16, fqShape, dataFqOutHiVec.data());

    const size_t dataFqLvl = 256;
    const auto dataFq =
            std::make_shared<ngraph::op::FakeQuantize>(producer, dataFqInLo, dataFqInHi, dataFqOutLo, dataFqOutHi,
                                                       dataFqLvl, ngraph::op::AutoBroadcastType::NUMPY);

    return dataFq->output(0);
}

class VPU3720Conv2dMixedPrecisionTest :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<std::tuple<bool, bool>> {
    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::U8);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
        cnnNetwork.getOutputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
    }
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice;
        const auto isInputQuantized = std::get<0>(GetParam());
        const auto isOutputQuantized = std::get<1>(GetParam());
        const InferenceEngine::SizeVector kenelShape = {16, 16, 3, 3};
        const size_t KERNEL_W = kenelShape.at(3);
        const size_t KERNEL_H = kenelShape.at(2);
        const size_t FILT_IN = kenelShape.at(1);
        const size_t FILT_OUT = kenelShape.at(0);
        const InferenceEngine::SizeVector inputShape = {1, 16, 32, 64};

        const auto params = ngraph::builder::makeParams(ngraph::element::f16, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const auto maybeQuantInput = quantize(paramOuts.at(0), FILT_IN, isInputQuantized);

        std::vector<ngraph::float16> weights(FILT_IN * FILT_OUT * KERNEL_W * KERNEL_H, 0.f);
        for (size_t i = 0; i < weights.size(); i++) {
            weights.at(i) = (1.f + std::sin(i * 3.14 / 6)) * 127.f;
        }
        const auto weightsLayer = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f16, ngraph::Shape{FILT_OUT, FILT_IN, KERNEL_H, KERNEL_W}, weights.data());
        const auto maybeQuantWeights = quantize(weightsLayer->output(0), 1, isInputQuantized);

        const auto convLayer = std::make_shared<ngraph::op::v1::Convolution>(
                maybeQuantInput, maybeQuantWeights, ngraph::Strides(std::vector<size_t>{1, 1}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ngraph::Strides(std::vector<size_t>{1, 1}));

        std::vector<ngraph::float16> biases(FILT_OUT, 1.0);
        for (size_t i = 0; i < biases.size(); i++) {
            biases.at(i) = i * 0.25f;
        }
        auto bias_weights_node = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f16, ngraph::Shape{1, FILT_OUT, 1, 1}, biases.data());

        auto bias_node = std::make_shared<ngraph::op::v1::Add>(convLayer->output(0), bias_weights_node->output(0));

        const auto maybeQuantOutput = quantize(bias_node->output(0), 1, isOutputQuantized);
        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(maybeQuantOutput)};

        function = std::make_shared<ngraph::Function>(results, params, "VPU3720Conv2dMixedPrecisionTest");

        threshold = 0.5f;
    }
};

TEST_P(VPU3720Conv2dMixedPrecisionTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

const std::vector<bool> isInputQuantized = {true, false};
const std::vector<bool> isOutputQuantized = {true, false};

INSTANTIATE_TEST_SUITE_P(conv2d_with_act, VPU3720Conv2dMixedPrecisionTest,
                         ::testing::Combine(::testing::ValuesIn(isInputQuantized),
                                            ::testing::ValuesIn(isOutputQuantized)));

}  // namespace
