// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "kmb_layer_test.hpp"
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>

namespace {

struct KmbScheduleSubGraphPrefetchTestParams {
    LayerTestsUtils::TargetDevice _device;
    InferenceEngine::SizeVector _in_dims;
    InferenceEngine::SizeVector _w_dims_conv_1;
    InferenceEngine::SizeVector _w_dims_conv_2;
    std::vector<uint64_t> _strides;
    std::vector<int64_t> _pads_begin;
    std::vector<int64_t> _pads_end;
    size_t N;
};

// Input -> -> Conv1 -> Conv2 -> ... -> Conv(N+3) -> Output

class KmbScheduleSubGraphPrefetchTest : public LayerTestsUtils::KmbLayerTestsCommon,
                                            public testing::WithParamInterface<KmbScheduleSubGraphPrefetchTestParams> {
    void SetUp() override {

        const auto test_params = GetParam();
        targetDevice = test_params._device;
        const InferenceEngine::SizeVector inputShape = test_params._in_dims;
        const InferenceEngine::SizeVector weights1Shape = test_params._w_dims_conv_1;
        const InferenceEngine::SizeVector weights2Shape = test_params._w_dims_conv_2;
        const InferenceEngine::SizeVector weights3Shape = {64, 32, 3, 3};
        const InferenceEngine::SizeVector weights4Shape = {128, 64, 3, 3};
        const InferenceEngine::SizeVector weights5Shape = {256, 128, 3, 3};
        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        
        const ngraph::Strides strides = test_params._strides;
        const ngraph::CoordinateDiff pads_begin = test_params._pads_begin;
        const ngraph::CoordinateDiff pads_end = test_params._pads_end;
        const ngraph::Strides dilations = {1, 1};
        std::vector<float> weights1(weights1Shape[0] * weights1Shape[1] * weights1Shape[2] * weights1Shape[3], 1);
        auto weights1FP32 = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, weights1Shape, weights1.data());
        
        const auto conv1 = std::make_shared<ngraph::opset2::Convolution>(paramOuts[0], weights1FP32, strides, pads_begin, pads_end, dilations);
        std::vector<float> weights2(weights2Shape[0] * weights2Shape[1] * weights2Shape[2] * weights2Shape[3], 1);
        auto weights2FP32 = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, weights2Shape, weights2.data());
        const auto conv2 = std::make_shared<ngraph::opset2::Convolution>(conv1, weights2FP32, strides, pads_begin, pads_end, dilations);
        
        std::vector<float> weights3(weights3Shape[0] * weights3Shape[1] * weights3Shape[2] * weights3Shape[3], 1);
        auto weights3FP32 = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, weights3Shape, weights3.data());
        const auto conv3 = std::make_shared<ngraph::opset2::Convolution>(conv2, weights3FP32, strides, pads_begin, pads_end, dilations);

        std::vector<float> weights4(weights4Shape[0] * weights4Shape[1] * weights4Shape[2] * weights4Shape[3], 1);
        auto weights4FP32 = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, weights4Shape, weights4.data());
        const auto conv4 = std::make_shared<ngraph::opset2::Convolution>(conv3, weights4FP32, strides, pads_begin, pads_end, dilations);

        std::vector<float> weights5(weights5Shape[0] * weights5Shape[1] * weights5Shape[2] * weights5Shape[3], 1);
        auto weights5FP32 = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, weights5Shape, weights5.data());
        const auto conv5 = std::make_shared<ngraph::opset2::Convolution>(conv4, weights5FP32, strides, pads_begin, pads_end, dilations);
        
        const ngraph::ResultVector results{
            std::make_shared<ngraph::opset1::Result>(conv5)
        };

        function = std::make_shared<ngraph::Function>(results, params, "KmbScheduleSubGraphPrefetchTest");
        threshold = 0.1f;
    }
};

TEST_P(KmbScheduleSubGraphPrefetchTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke, KmbScheduleSubGraphPrefetchTest,
    ::testing::Values(
    KmbScheduleSubGraphPrefetchTestParams {
        LayerTestsUtils::testPlatformTargetDevice,  // _device
        {1, 1, 64, 64},   // in dims
        {16, 1, 3, 3},    // weights 1 dims
        {32, 16, 3, 3},    // weights 2 dims
        {2, 2},            // strides
        {0, 0},            // pads_begin
        {0, 0},            // pads_end
        10,                // N
    })
);

}