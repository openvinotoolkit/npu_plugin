// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "kmb_layer_test.hpp"
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>

namespace {

struct KmbScheduleSubGraphPrefetchMatMulTestParams {
    LayerTestsUtils::TargetDevice _device;
    InferenceEngine::SizeVector _in_dims;
    InferenceEngine::SizeVector _w_dims_conv_1;
    InferenceEngine::SizeVector _w_dims_conv_2;
    std::vector<uint64_t> _strides;
    std::vector<int64_t> _pads_begin;
    std::vector<int64_t> _pads_end;
};

// Input -> Conv1 -> Add -> Permute -> Permute -> Reshape -> Permute -> Conv(MatMul) -> Output

class KmbScheduleSubGraphPrefetchMatMulTest : public LayerTestsUtils::KmbLayerTestsCommon,
                                        public testing::WithParamInterface<KmbScheduleSubGraphPrefetchMatMulTestParams> {
    void SetUp() override {

        const auto test_params = GetParam();
        targetDevice = test_params._device;
        const InferenceEngine::SizeVector inputShape = test_params._in_dims;
        const InferenceEngine::SizeVector weights1Shape = test_params._w_dims_conv_1;
        const InferenceEngine::SizeVector weights2Shape = test_params._w_dims_conv_2;
        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const ngraph::Strides strides = test_params._strides;
        const ngraph::CoordinateDiff pads_begin = test_params._pads_begin;
        const ngraph::CoordinateDiff pads_end = test_params._pads_end;
        const ngraph::Strides dilations = {1, 1};

        // 1. Conv: input[1x128x4x4],NHWC filter[256x128x3x3] output 1x256x2x2
        std::vector<float> weights1(weights1Shape[0] * weights1Shape[1] * weights1Shape[2] * weights1Shape[3], 1);
        auto weights1FP32 = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, weights1Shape, weights1.data());
        const auto conv1 = std::make_shared<ngraph::opset2::Convolution>(paramOuts[0], weights1FP32, strides, pads_begin, pads_end, dilations);

        // 2. Add: input1 conv1 [1x256x2x2] NHWC, input2 conv1 [1x256x2x2]
        const auto add2 = std::make_shared<ngraph::opset1::Add>(conv1, conv1);

        // 3. Permute: input [1x256x2x2] NHWC, output [1x256x2x2] NCHW
        const std::vector<int64_t> permute3_order({0, 1, 2, 3});
        const auto permute3_order_arg = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ov::Shape({permute3_order.size()}), permute3_order);
        const auto permute3 = std::make_shared<ngraph::opset2::Transpose>(add2, permute3_order_arg);

        // 4. Permute: input [1x256x2x2] NCHW, output [1x2x2x256] NCHW
        const std::vector<int64_t> permute4_order({0, 2, 3, 1});
        const auto permute4_order_arg = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ov::Shape({permute4_order.size()}), permute4_order);
        const auto permute4 = std::make_shared<ngraph::opset2::Transpose>(permute3, permute4_order_arg);

        // 5. Reshape: input [1x2x2x256] NCHW, output [1x1024x1x1] NCHW
        const std::vector<int64_t> reshape5_shape({1, 1024, 1, 1});
        const auto reshape5_shape_arg = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ov::Shape({reshape5_shape.size()}), reshape5_shape);
        const auto reshape5 = std::make_shared<ngraph::opset2::Reshape>(permute4, reshape5_shape_arg, false);

        // 6. Permute: input [1x1024x1x1] NCHW, output [1x1024x1x1] NHWC
        const std::vector<int64_t> permute6_order({0, 1, 2, 3});
        const auto permute6_order_arg = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ov::Shape({permute6_order.size()}), permute6_order);
        const auto permute6 = std::make_shared<ngraph::opset2::Transpose>(reshape5, permute6_order_arg);

        // 7. MatMul Conv: input [1x1024x1x1] NHWC, filter [256x1024x1x1] output [1x256x1x1]
        std::vector<float> weights2(weights2Shape[0] * weights2Shape[1] * weights2Shape[2] * weights2Shape[3], 1);
        auto weights2FP32 = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, weights2Shape, weights2.data());
        const auto conv7 = std::make_shared<ngraph::opset2::Convolution>(permute6, weights2FP32, strides, pads_begin, pads_end, dilations);

        const ngraph::ResultVector results{
                std::make_shared<ngraph::opset1::Result>(conv7)
        };

        function = std::make_shared<ngraph::Function>(results, params, "KmbScheduleSubGraphPrefetchMatMulTest");
        threshold = 0.1f;
    }
};

TEST_P(KmbScheduleSubGraphPrefetchMatMulTest, CompareWithRefs_MLIR) {
useCompilerMLIR();
setDefaultHardwareModeMLIR();
Run();
}

INSTANTIATE_TEST_CASE_P(smoke, KmbScheduleSubGraphPrefetchMatMulTest,
        ::testing::Values(
        KmbScheduleSubGraphPrefetchMatMulTestParams {
                LayerTestsUtils::testPlatformTargetDevice,  // _device
                {1, 128, 4, 4},   // in dims
                {256, 128, 3, 3},    // weights 1 dims
                {256, 1024, 1, 1},    // weights 2 dims
                {1, 1},            // strides
                {0, 0},            // pads_begin
                {0, 0},            // pads_end
        })
);
}
