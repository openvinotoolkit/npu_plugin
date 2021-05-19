// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class KmbConv2dWithBiasTest : public LayerTestsUtils::KmbLayerTestsCommon {
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice;
        const size_t KERNEL_W = 2;
        const size_t KERNEL_H = 2;
        const size_t FILT_IN = 16;
        const size_t FILT_OUT = 16;
        const InferenceEngine::SizeVector inputShape = {1, FILT_IN, 4, 4};

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

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(bias_node)};

        function = std::make_shared<ngraph::Function>(results, params, "KmbConv2dWithBiasTest");

        threshold = 1.5f;
    }
};

TEST_F(KmbConv2dWithBiasTest, CompareWithRefs_MCM) {
    Run();
}

TEST_F(KmbConv2dWithBiasTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    Run();
}

TEST_F(KmbConv2dWithBiasTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setReferenceHardwareModeMLIR();
    Run();
}

}  // namespace
