//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpu_ov1_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

std::shared_ptr<ov::Node> buildConvolution(const ov::Output<ov::Node>& param, const size_t filtersIn,
                                           const size_t filtersOut) {
    const size_t kernelW = 1;
    const size_t kernelH = 1;
    std::vector<float> weights(filtersOut * filtersIn * kernelW * kernelH);
    for (std::size_t i = 0; i < weights.size(); i++) {
        weights.at(i) = std::cos(i * 3.14 / 6);
    }
    auto constLayerNode = std::make_shared<ngraph::op::Constant>(
            ngraph::element::Type_t::f32, ngraph::Shape{filtersOut, filtersIn, kernelH, kernelW}, weights.data());

    auto conv2d = std::make_shared<ngraph::op::v1::Convolution>(
            param, constLayerNode->output(0), ngraph::Strides(std::vector<size_t>{1, 1}),
            ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
            ngraph::Strides(std::vector<size_t>{1, 1}));

    return conv2d;
}

class VPUXTilingWithConcatTest_VPU3720 :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
        public testing::WithParamInterface<InferenceEngine::SizeVector> {
    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
        cnnNetwork.getOutputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
    }
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice();
        const auto inputShape = GetParam();
        const size_t filtIn = inputShape.at(1);
        const size_t filtOut = 64;

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const auto conv2d64Planes = buildConvolution(paramOuts.at(0), filtIn, filtOut);
        const auto conv2d32Planes = buildConvolution(conv2d64Planes->output(0), filtOut, 32);

        const auto concat = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector({conv2d64Planes->output(0), conv2d32Planes->output(0)}), 1);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};

        function = std::make_shared<ngraph::Function>(results, params, "VPUXTilingWithConcatTest");

        threshold = 0.5f;
    }
};

TEST_P(VPUXTilingWithConcatTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

const std::vector<InferenceEngine::SizeVector> inputShapes = {
        {1, 16, 175, 175},
        {1, 3, 250, 250},
};

INSTANTIATE_TEST_SUITE_P(smoke_tiling_with_concat, VPUXTilingWithConcatTest_VPU3720, ::testing::ValuesIn(inputShapes));

}  // namespace
