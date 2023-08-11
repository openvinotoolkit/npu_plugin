// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class VPUXConv2dWithTransposeTest_VPU3720 :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<std::tuple<std::vector<int64_t>, InferenceEngine::Layout>> {
    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
        const auto outLayout = std::get<1>(GetParam());
        cnnNetwork.getOutputsInfo().begin()->second->setLayout(outLayout);
    }
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice;
        const auto transposeOrder = std::get<0>(GetParam());
        const InferenceEngine::SizeVector lhsInputShape = {1, 3, 32, 64};

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {lhsInputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const auto add = buildConv(paramOuts.at(0));
        const auto transpose = buildTranspose(add->output(0), transposeOrder);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset7::Result>(transpose)};

        function = std::make_shared<ngraph::Function>(results, params, "VPUXConv2dWithTransposeTest");

        threshold = 0.5f;
    }

    std::shared_ptr<ov::Node> buildConv(const ov::Output<ov::Node>& param) {
        const InferenceEngine::SizeVector inputShape = param.get_shape();
        const auto weightsSize = inputShape.at(1) * 16 * 1 * 1;
        std::vector<float> values(weightsSize, 1.f);
        const auto weightsShape = ngraph::Shape{16, inputShape.at(1), 1, 1};
        const auto weights = ngraph::opset8::Constant::create(ngraph::element::f32, weightsShape, values);
        auto conv2d_node = std::make_shared<ngraph::op::v1::Convolution>(
                param, weights->output(0), ngraph::Strides(std::vector<size_t>{1, 1}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ngraph::Strides(std::vector<size_t>{1, 1}));

        return conv2d_node;
    }

    std::shared_ptr<ov::Node> buildTranspose(const ov::Output<ov::Node>& param, const std::vector<int64_t>& dimsOrder) {
        auto order = ngraph::opset8::Constant::create(ngraph::element::i64, {dimsOrder.size()}, dimsOrder);
        return std::make_shared<ngraph::opset7::Transpose>(param, order);
    }
};

TEST_P(VPUXConv2dWithTransposeTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

const std::vector<std::vector<int64_t>> transposes = {
        {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
};

const std::vector<InferenceEngine::Layout> outLayout = {InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC};

INSTANTIATE_TEST_SUITE_P(smoke_transposeConv2d, VPUXConv2dWithTransposeTest_VPU3720,
                         ::testing::Combine(::testing::ValuesIn(transposes), ::testing::ValuesIn(outLayout)));

}  // namespace
