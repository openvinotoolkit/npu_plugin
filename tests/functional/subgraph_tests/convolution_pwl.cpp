// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

enum class PostOp {
    SIGMOID,
    TANH,
    LRELU
};

class KmbConvPwlSubGraphTest :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<std::tuple<LayerTestsUtils::TargetDevice, PostOp>> {
    void SetUp() override {
        const InferenceEngine::SizeVector inputShape{1, 3, 32, 32};
        const InferenceEngine::SizeVector weightsShape{16, 3, 1, 1};

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const auto weightsU8 =
                ngraph::builder::makeConstant<uint8_t>(ngraph::element::u8, weightsShape, {}, true, 255, 0);
        const auto weightsFP32 = std::make_shared<ngraph::opset2::Convert>(weightsU8, ngraph::element::f32);

        const ngraph::Strides strides = {1, 1};
        const ngraph::CoordinateDiff pads_begin = {0, 0};
        const ngraph::CoordinateDiff pads_end = {0, 0};
        const ngraph::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ngraph::opset2::Convolution>(paramOuts[0], weightsFP32, strides, pads_begin,
                                                                        pads_end, dilations);

        std::shared_ptr<ngraph::Node> postOp;
        auto postOpType = std::get<1>(GetParam());
        if (postOpType == PostOp::SIGMOID) {
            postOp = std::make_shared<ngraph::opset7::Sigmoid>(conv);
        } else if (postOpType == PostOp::TANH) {
            postOp = std::make_shared<ngraph::opset7::Tanh>(conv);
        } else if (postOpType == PostOp::LRELU) {
            const auto negativeSlope = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1}, {0.01}, false);
            postOp = std::make_shared<ngraph::opset7::PRelu>(conv, negativeSlope);
        }

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(postOp)};
        function = std::make_shared<ngraph::Function>(results, params, "KmbConvPwl");

        targetDevice = std::get<0>(GetParam());
        threshold = 0.1f;
    }
};

class KmbConvPwlQuantizedSubGraphTest :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<std::tuple<LayerTestsUtils::TargetDevice, PostOp>> {
    void SetUp() override {
        const InferenceEngine::SizeVector inputShape{1, 3, 32, 32};
        const InferenceEngine::SizeVector weightsShape{16, 3, 1, 1};

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const size_t dataLevels = 256;
        const auto dataFq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngraph::element::f32, dataLevels, {},
                                                            {-3.0}, {3.0}, {-3.0}, {3.0});

        const auto weightsU8 =
                ngraph::builder::makeConstant<uint8_t>(ngraph::element::u8, weightsShape, {}, true, 255, 0);
        const auto weightsFP32 = std::make_shared<ngraph::opset2::Convert>(weightsU8, ngraph::element::f32);

        const auto weightsInLow = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1}, {0.0f}, false);
        const auto weightsInHigh = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1}, {255.0f}, false);
        std::vector<float> perChannelLow(weightsShape[0], 0.0f);
        std::vector<float> perChannelHigh(weightsShape[0], 1.0f);
        const auto weightsOutLow = ngraph::builder::makeConstant<float>(
                ngraph::element::f32, {weightsShape[0], 1, 1, 1}, perChannelLow, false);
        const auto weightsOutHigh = ngraph::builder::makeConstant<float>(
                ngraph::element::f32, {weightsShape[0], 1, 1, 1}, perChannelHigh, false);

        const size_t weightsLevels = 256;
        const auto weightsFq = std::make_shared<ngraph::opset2::FakeQuantize>(
                weightsFP32, weightsInLow, weightsInHigh, weightsOutLow, weightsOutHigh, weightsLevels);

        const ngraph::Strides strides = {1, 1};
        const ngraph::CoordinateDiff pads_begin = {0, 0};
        const ngraph::CoordinateDiff pads_end = {0, 0};
        const ngraph::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ngraph::opset2::Convolution>(dataFq, weightsFq, strides, pads_begin,
                                                                        pads_end, dilations);

        std::shared_ptr<ngraph::Node> outputFq;
        const size_t outLevels = 256;
        auto postOpType = std::get<1>(GetParam());
        if (postOpType == PostOp::SIGMOID) {
            const auto postOp = std::make_shared<ngraph::opset7::Sigmoid>(conv);
            outputFq = ngraph::builder::makeFakeQuantize(postOp, ngraph::element::f32, outLevels, {},
                                                         {0.0}, {1.0}, {0.0}, {1.0});
        } else if (postOpType == PostOp::TANH) {
            const auto postOp = std::make_shared<ngraph::opset7::Tanh>(conv);
            outputFq = ngraph::builder::makeFakeQuantize(postOp, ngraph::element::f32, outLevels, {},
                                                         {-1.0}, {1.0}, {-1.0}, {1.0});
        } else if (postOpType == PostOp::LRELU) {
            const auto negativeSlope = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1}, {0.01}, false);
            const auto postOp = std::make_shared<ngraph::op::v0::PRelu>(conv, negativeSlope);
            outputFq = ngraph::builder::makeFakeQuantize(postOp, ngraph::element::f32, outLevels, {},
                                                         {-128.0}, {127.0}, {-128.0}, {127.0}); // need to check with Harald
        }

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(outputFq)};
        function = std::make_shared<ngraph::Function>(results, params, "KmbConvPwlQuantized");

        targetDevice = std::get<0>(GetParam());
        threshold = 0.1f;
    }
};

TEST_P(KmbConvPwlSubGraphTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbConvPwlSubGraphTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(KmbConvPwlQuantizedSubGraphTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbConvPwlQuantizedSubGraphTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

std::vector<PostOp> postOps = {
    PostOp::SIGMOID, PostOp::TANH, PostOp::LRELU
};

INSTANTIATE_TEST_CASE_P(smoke, KmbConvPwlSubGraphTest,
                        ::testing::Combine(
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::ValuesIn(postOps)));

INSTANTIATE_TEST_CASE_P(smoke, KmbConvPwlQuantizedSubGraphTest,
                        ::testing::Combine(
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::ValuesIn(postOps)));

}  // namespace
