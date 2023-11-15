//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpu_ov1_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

struct ConvSoftmaxConvTestParams {
    InferenceEngine::SizeVector inputShape;
    int64_t axis;
    size_t softmaxChannelSize;
};

class VPUXConvSoftmaxConvTest :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
        public testing::WithParamInterface<ConvSoftmaxConvTestParams> {
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice();
        inLayout = InferenceEngine::Layout::NHWC;
        outLayout = InferenceEngine::Layout::NHWC;
        inPrc = InferenceEngine::Precision::FP16;
        outPrc = InferenceEngine::Precision::FP16;
        const auto testParams = GetParam();
        int64_t axis = testParams.axis;
        const auto inputShape = testParams.inputShape;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inPrc);
        auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
        auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        const auto conv1 = buildConv(paramOuts[0], testParams.softmaxChannelSize);
        const auto softMax = std::make_shared<ngraph::opset1::Softmax>(conv1->output(0), axis);
        const auto conv2 = buildConv(softMax->output(0), inputShape.at(1));
        const ngraph::ResultVector results{std::make_shared<ngraph::opset7::Result>(conv2)};
        function = std::make_shared<ngraph::Function>(results, params, "VPUXConvSoftmaxConvTest");
        abs_threshold = 0.008;
    }

    std::shared_ptr<ov::Node> buildConv(const ov::Output<ov::Node>& param, size_t softmaxChannelSize) {
        const InferenceEngine::SizeVector inputShape = param.get_shape();
        const auto weightsSize = inputShape.at(1) * softmaxChannelSize * 1 * 1;
        std::vector<float> values(weightsSize, 0.0f);
        for (std::size_t i = 0; i < softmaxChannelSize; i++) {
            values.at(i * inputShape.at(1) + i % inputShape.at(1)) = 1.0f;
        }
        const auto weightsShape = ngraph::Shape{softmaxChannelSize, inputShape.at(1), 1, 1};
        const auto weights = ngraph::opset8::Constant::create(ngraph::element::f16, weightsShape, values);
        auto conv2d_node = std::make_shared<ngraph::op::v1::Convolution>(
                param, weights->output(0), ngraph::Strides(std::vector<size_t>{1, 1}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ngraph::Strides(std::vector<size_t>{1, 1}));

        return conv2d_node;
    }
};

TEST_P(VPUXConvSoftmaxConvTest, VPU3720_HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_ConvSoftmaxConv, VPUXConvSoftmaxConvTest,
                         ::testing::Values(ConvSoftmaxConvTestParams{{1, 48, 8, 16}, 1, 77}));

}  // namespace
