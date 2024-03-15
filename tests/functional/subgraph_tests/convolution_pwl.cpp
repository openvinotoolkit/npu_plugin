// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <ov_models/builders.hpp>
#include <ov_models/utils/ov_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace ov::test {

enum class PostOp { SIGMOID, TANH, PRELU };

class ConvPwlSubGraphTest_NPU3700 : public VpuOv2LayerTest, public testing::WithParamInterface<PostOp> {
    void SetUp() override {
        const ov::Shape inputShape{1, 3, 32, 32};
        const ov::Shape weightsShape{16, 3, 1, 1};

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(inputShape))};

        const auto weightsU8 = ngraph::builder::makeConstant<uint8_t>(ov::element::u8, weightsShape, {}, true, 255, 0);
        const auto weightsFP32 = std::make_shared<ov::op::v0::Convert>(weightsU8, ov::element::f32);

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ov::op::v1::Convolution>(params[0], weightsFP32, strides, pads_begin,
                                                                    pads_end, dilations);

        std::shared_ptr<ov::Node> postOp;
        auto postOpType = GetParam();
        if (postOpType == PostOp::SIGMOID) {
            postOp = std::make_shared<ov::op::v0::Sigmoid>(conv);
        } else if (postOpType == PostOp::TANH) {
            postOp = std::make_shared<ov::op::v0::Tanh>(conv);
        } else if (postOpType == PostOp::PRELU) {
            const auto negativeSlope = ngraph::builder::makeConstant<float>(ov::element::f32, {1}, {0.1}, false);
            postOp = std::make_shared<ov::op::v0::PRelu>(conv, negativeSlope);
        }

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(postOp)};
        function = std::make_shared<ov::Model>(results, params, "ConvPwl");
        rel_threshold = 0.1f;
    }
};

class ConvPwlQuantizedSubGraphTest_NPU3700 : public VpuOv2LayerTest, public testing::WithParamInterface<PostOp> {
    void SetUp() override {
        const ov::Shape inputShape{1, 3, 32, 32};
        const ov::Shape weightsShape{16, 3, 1, 1};

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        const size_t dataLevels = 256;
        const auto dataFq = ngraph::builder::makeFakeQuantize(params[0], ov::element::f32, dataLevels, {}, {-3.0},
                                                              {3.0}, {-3.0}, {3.0});

        const auto weightsU8 = ngraph::builder::makeConstant<uint8_t>(ov::element::u8, weightsShape, {}, true, 255, 0);
        const auto weightsFP32 = std::make_shared<ov::op::v0::Convert>(weightsU8, ov::element::f32);

        const auto weightsInLow = ngraph::builder::makeConstant<float>(ov::element::f32, {1}, {0.0f}, false);
        const auto weightsInHigh = ngraph::builder::makeConstant<float>(ov::element::f32, {1}, {255.0f}, false);
        std::vector<float> perChannelLow(weightsShape[0], 0.0f);
        std::vector<float> perChannelHigh(weightsShape[0], 1.0f);
        const auto weightsOutLow = ngraph::builder::makeConstant<float>(ov::element::f32, {weightsShape[0], 1, 1, 1},
                                                                        perChannelLow, false);
        const auto weightsOutHigh = ngraph::builder::makeConstant<float>(ov::element::f32, {weightsShape[0], 1, 1, 1},
                                                                         perChannelHigh, false);

        const size_t weightsLevels = 256;
        const auto weightsFq = std::make_shared<ov::op::v0::FakeQuantize>(weightsFP32, weightsInLow, weightsInHigh,
                                                                          weightsOutLow, weightsOutHigh, weightsLevels);

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        const auto conv =
                std::make_shared<ov::op::v1::Convolution>(dataFq, weightsFq, strides, pads_begin, pads_end, dilations);

        std::shared_ptr<ov::Node> outputFq;
        const size_t outLevels = 256;
        auto postOpType = GetParam();
        if (postOpType == PostOp::SIGMOID) {
            const auto postOp = std::make_shared<ov::op::v0::Sigmoid>(conv);
            outputFq = ngraph::builder::makeFakeQuantize(postOp, ov::element::f32, outLevels, {}, {0.0}, {1.0}, {0.0},
                                                         {1.0});
        } else if (postOpType == PostOp::TANH) {
            const auto postOp = std::make_shared<ov::op::v0::Tanh>(conv);
            outputFq = ngraph::builder::makeFakeQuantize(postOp, ov::element::f32, outLevels, {}, {-1.0}, {1.0}, {-1.0},
                                                         {1.0});
        } else if (postOpType == PostOp::PRELU) {
            const auto negativeSlope = ngraph::builder::makeConstant<float>(ov::element::f32, {1}, {0.1}, false);
            const auto postOp = std::make_shared<ov::op::v0::PRelu>(conv, negativeSlope);
            outputFq = ngraph::builder::makeFakeQuantize(postOp, ov::element::f32, outLevels, {}, {-128.0}, {127.0},
                                                         {-128.0}, {127.0});
        }

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outputFq)};
        function = std::make_shared<ov::Model>(results, params, "ConvPwlQuantized");
        rel_threshold = 0.1f;
    }
};

TEST_P(ConvPwlSubGraphTest_NPU3700, SW) {
    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3700);
}

TEST_P(ConvPwlSubGraphTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3700);
}

TEST_P(ConvPwlQuantizedSubGraphTest_NPU3700, SW) {
    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3700);
}

TEST_P(ConvPwlQuantizedSubGraphTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3700);
}

std::vector<PostOp> postOps = {PostOp::SIGMOID, PostOp::TANH, PostOp::PRELU};

// TODO: investigate bad accuracy for both SW and HW
// prelu quantized test
std::vector<PostOp> quantPostOps = {
        PostOp::SIGMOID, PostOp::TANH
        //, PostOp::PRELU
};

INSTANTIATE_TEST_CASE_P(smoke_ConvPwl, ConvPwlSubGraphTest_NPU3700, ::testing::ValuesIn(postOps));

INSTANTIATE_TEST_CASE_P(smoke_ConvPwlQuantized, ConvPwlQuantizedSubGraphTest_NPU3700,
                        ::testing::ValuesIn(quantPostOps));

}  // namespace ov::test
