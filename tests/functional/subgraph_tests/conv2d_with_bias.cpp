// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

std::shared_ptr<ngraph::Node> build_activation_helper(const std::shared_ptr<ngraph::Node>& producer,
                                                      const ngraph::helpers::ActivationTypes& act_type) {
    switch (act_type) {
    case ngraph::helpers::None:
        return producer;
    case ngraph::helpers::Relu:
        return std::make_shared<ngraph::op::Relu>(producer->output(0));
    default:
        IE_THROW() << "build_activation_helper: unsupported activation type: " << act_type;
    }

    // execution must never reach this point
    return nullptr;
}

class KmbConv2dWithBiasTest :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector, ngraph::helpers::ActivationTypes>> {
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice;
        const auto kenelShape = std::get<0>(GetParam());
        const size_t KERNEL_W = kenelShape.at(3);
        const size_t KERNEL_H = kenelShape.at(2);
        const size_t FILT_IN = kenelShape.at(1);
        const size_t FILT_OUT = kenelShape.at(0);
        const InferenceEngine::SizeVector inputShape = {1, FILT_IN, KERNEL_H * 2, KERNEL_W * 2};

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        std::vector<float> weights(FILT_IN * FILT_OUT * KERNEL_W * KERNEL_H);
        for (int i = 0; i < weights.size(); i++) {
            weights.at(i) = std::cos(i * 3.14 / 6);
        }
        auto constLayer_node = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, ngraph::Shape{FILT_OUT, FILT_IN, KERNEL_H, KERNEL_W}, weights.data());

        auto conv2d_node = std::make_shared<ngraph::op::v1::Convolution>(
                paramOuts.at(0), constLayer_node->output(0), ngraph::Strides(std::vector<size_t>{1, 1}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ngraph::Strides(std::vector<size_t>{1, 1}));

        std::vector<float> biases(FILT_OUT, 1.0);
        for (int i = 0; i < biases.size(); i++) {
            biases.at(i) = i * 0.25f;
        }
        auto bias_weights_node = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, ngraph::Shape{1, FILT_OUT, 1, 1}, biases.data());

        auto bias_node = std::make_shared<ngraph::op::v1::Add>(conv2d_node->output(0), bias_weights_node->output(0));
        auto act_type = std::get<1>(GetParam());
        auto act_node = build_activation_helper(bias_node, act_type);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(act_node)};

        function = std::make_shared<ngraph::Function>(results, params, "KmbConv2dWithBiasTest");

        threshold = 0.5f;
    }
};

TEST_P(KmbConv2dWithBiasTest, CompareWithRefs_MCM) {
    Run();
}

TEST_P(KmbConv2dWithBiasTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbConv2dWithBiasTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setReferenceHardwareModeMLIR();
    Run();
}

const std::vector<InferenceEngine::SizeVector> kernelShapes = {
        {11, 3, 2, 2},
        {16, 3, 2, 2},
        {11, 16, 2, 2},
        {16, 16, 2, 2},
};

const std::vector<ngraph::helpers::ActivationTypes> activations = {
        ngraph::helpers::None,
        ngraph::helpers::Relu,
};

INSTANTIATE_TEST_CASE_P(conv2d_with_act, KmbConv2dWithBiasTest,
                        ::testing::Combine(::testing::ValuesIn(kernelShapes), ::testing::ValuesIn(activations)));

}  // namespace
