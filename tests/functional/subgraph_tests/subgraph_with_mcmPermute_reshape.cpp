// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>

namespace {

struct KmbScheduleSubGraphMcmPermuteWithReshapeTestParams {
    LayerTestsUtils::TargetDevice _device;
    InferenceEngine::SizeVector _in_dims;
    InferenceEngine::SizeVector _w_dims_conv_1;
    InferenceEngine::SizeVector _w_dims_conv_2;
    std::vector<uint64_t> _strides;
    std::vector<int64_t> _pads_begin;
    std::vector<int64_t> _pads_end;
    std::vector<int64_t> _transpose_order_1;
    std::vector<int64_t> _transpose_order_2;
};

// Create a subgraph to debug mcmPermute 
//
// Input -> Conv1 -> Transpose1 -> Reshape -> Transpose2 -> Conv2 -> Output
// 
class KmbScheduleSubGraphMcmPermuteWithReshapeTest : public LayerTestsUtils::KmbLayerTestsCommon,
                                            public testing::WithParamInterface<KmbScheduleSubGraphMcmPermuteWithReshapeTestParams> {
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

        // Conv1
        std::vector<float> weights1(weights1Shape[0] * weights1Shape[1] * weights1Shape[2] * weights1Shape[3], 1);
        auto weights1FP32 = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, weights1Shape, weights1.data());

        const auto conv1 = std::make_shared<ngraph::opset2::Convolution>(paramOuts[0], weights1FP32, strides, pads_begin, pads_end, dilations);

        // Transpose1
        const auto input_order1 = test_params._transpose_order_1;
        const auto new_Shape_order1 =
                ngraph::builder::makeConstant<int64_t>(ngraph::element::i64, ngraph::Shape{4}, input_order1, false);
        const auto transpose1 = std::make_shared<ngraph::opset2::Transpose>(conv1, new_Shape_order1);

        // Reshape
        const auto reshapePattern =
                std::vector<int64_t>({static_cast<int64_t>(transpose1->get_output_shape(0)[0]), 
                                    static_cast<int64_t>(transpose1->get_output_shape(0)[1] * 2), 
                                    static_cast<int64_t>( transpose1->get_output_shape(0)[2]), 
                                    static_cast<int64_t>( transpose1->get_output_shape(0)[3] / 2)});
        const auto newShape =
                ngraph::builder::makeConstant<int64_t>(ngraph::element::i64, ngraph::Shape{4}, reshapePattern, false);
        const auto reshape1 = std::make_shared<ngraph::op::v1::Reshape>(transpose1, newShape, true);

        // Transpose2
        const auto input_order2 = test_params._transpose_order_2;
        const auto new_Shape_order2 =
                ngraph::builder::makeConstant<int64_t>(ngraph::element::i64, ngraph::Shape{4}, input_order2, false);
        const auto transpose2 = std::make_shared<ngraph::opset2::Transpose>(reshape1, new_Shape_order2);

        // Conv2
        std::vector<float> weights2(weights2Shape[0] * weights2Shape[1] * weights2Shape[2] * weights2Shape[3], 1);
        auto weights2FP32 = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, weights2Shape, weights2.data());

        const auto conv2 = std::make_shared<ngraph::opset2::Convolution>(transpose2, weights2FP32, strides, pads_begin, pads_end, dilations);

        const ngraph::ResultVector results{
            std::make_shared<ngraph::opset1::Result>(conv2)
        };
        function = std::make_shared<ngraph::Function>(results, params, "KmbScheduleSubGraphMcmPermuteWithReshapeTest");

        threshold = 0.1f;
    }
};

TEST_P(KmbScheduleSubGraphMcmPermuteWithReshapeTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke, KmbScheduleSubGraphMcmPermuteWithReshapeTest,
    ::testing::Values(
    KmbScheduleSubGraphMcmPermuteWithReshapeTestParams {
        LayerTestsUtils::testPlatformTargetDevice,  // _device
        {1, 16, 2, 1},   // in dims
        {512, 16, 1, 1},    // weights 1 dims
        {16, 256, 1, 1},    // weights 2 dims
        {1, 1},            // strides
        {0, 0},            // pads_begin
        {0, 0},            // pads_end
        {0, 2, 3, 1},      // _transpose_order_1
        {0, 3, 1, 2},      // _transpose_order_2
    })
);

}
