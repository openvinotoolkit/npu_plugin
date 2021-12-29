// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>

namespace {

struct KmbScheduleSubGraphMcmPermuteWithSliceTestParams {
    LayerTestsUtils::TargetDevice _device;
    InferenceEngine::SizeVector _in_dims;
    InferenceEngine::SizeVector _w_dims_conv_1;
    InferenceEngine::SizeVector _w_dims_conv_2;
    std::vector<uint64_t> _strides;
    std::vector<int64_t> _pads_begin;
    std::vector<int64_t> _pads_end;

    std::vector<int64_t> _begin_data;
    std::vector<int64_t> _end_data;
    std::vector<int64_t> _strides_data;
};

// Create a subgraph to debug mcmPermute with slice
// 
// Input -> Conv1 -> Slice -> Conv2 -> Output
// 
class KmbScheduleSubGraphMcmPermuteWithSliceTest : public LayerTestsUtils::KmbLayerTestsCommon,
                                            public testing::WithParamInterface<KmbScheduleSubGraphMcmPermuteWithSliceTestParams> {
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

        // Slice
        std::vector<int64_t> sliceBegin = test_params._begin_data;
        const auto sliceBeginConst = ngraph::builder::makeConstant<int64_t>(ngraph::element::i64, {sliceBegin.size()}, sliceBegin, false);
        std::vector<int64_t> sliceEnd = test_params._end_data;
        const auto sliceEndConst = ngraph::builder::makeConstant<int64_t>(ngraph::element::i64, {sliceEnd.size()}, sliceEnd, false);
        std::vector<int64_t> sliceStrides = test_params._strides_data;
        const auto sliceStridesConst = ngraph::builder::makeConstant<int64_t>(ngraph::element::i64, {sliceStrides.size()}, sliceStrides, false);

        const auto sliceOp =
            std::make_shared<ngraph::op::v8::Slice>(conv1,
                                                    sliceBeginConst,
                                                    sliceEndConst,
                                                    sliceStridesConst);

        // Conv2
        std::vector<float> weights2(weights2Shape[0] * weights2Shape[1] * weights2Shape[2] * weights2Shape[3], 1);
        auto weights2FP32 = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, weights2Shape, weights2.data());

        const auto conv2 = std::make_shared<ngraph::opset2::Convolution>(sliceOp, weights2FP32, strides, pads_begin, pads_end, dilations);

        const ngraph::ResultVector results{
            std::make_shared<ngraph::opset1::Result>(conv2)
        };
        function = std::make_shared<ngraph::Function>(results, params, "KmbScheduleSubGraphMcmPermuteWithSliceTest");

        threshold = 0.1f;
    }
};

TEST_P(KmbScheduleSubGraphMcmPermuteWithSliceTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke, KmbScheduleSubGraphMcmPermuteWithSliceTest,
    ::testing::Values(
    KmbScheduleSubGraphMcmPermuteWithSliceTestParams {
        LayerTestsUtils::testPlatformTargetDevice,  // _device
        {1, 16, 80, 80},   // in dims
        {32, 16, 1, 1},    // weights 1 dims
        {16, 32, 1, 1},    // weights 2 dims
        {1, 1},            // strides
        {0, 0},            // pads_begin
        {0, 0},            // pads_end

        {0, 0, 0, 0},      // begin data
        {1, 32, 80, 80},    // end data
        {1, 1, 2, 2},      // strides data
    })
);

}
