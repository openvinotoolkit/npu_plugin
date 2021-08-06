// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>

namespace {

struct KmbAsymmetricStrideConvSubGraphTestParams {
    LayerTestsUtils::TargetDevice _device;
    InferenceEngine::SizeVector _in_dims;
    InferenceEngine::SizeVector _w_dims;
    std::vector<uint64_t> _strides;
    std::vector<int64_t> _pads_begin;
    std::vector<int64_t> _pads_end;
};

class KmbAsymmetricStrideConvSubGraphTest : public LayerTestsUtils::KmbLayerTestsCommon,
                                            public testing::WithParamInterface<KmbAsymmetricStrideConvSubGraphTestParams> {
    void SetUp() override {
        const auto test_params = GetParam();
        targetDevice = test_params._device;
        const InferenceEngine::SizeVector inputShape = test_params._in_dims;
        const InferenceEngine::SizeVector weightsShape = test_params._w_dims;

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const size_t dataLevels = 256;
        const std::vector<float> dataLow = {0.0f};
        const std::vector<float> dataHigh = {255.0f};
        const auto dataFq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngraph::element::f32, dataLevels, {}, dataLow, dataHigh, dataLow, dataHigh);

        std::vector<uint64_t> poolStridesVec = {1, 1};
        std::vector<uint64_t> poolKernelVec = {1, 1};
        const ngraph::Strides poolStrides = poolStridesVec;
        const ngraph::Shape padsBegin = {0, 0};
        const ngraph::Shape padsEnd = {0, 0};
        const ngraph::Shape poolKernel = poolKernelVec;
        const auto pool = std::make_shared<ngraph::opset2::MaxPool>(dataFq, poolStrides, padsBegin, padsEnd, poolKernel);

        std::vector<float> weights(weightsShape[0] * weightsShape[1] * weightsShape[2] * weightsShape[3]);
        for (std::size_t i = 0; i < weights.size(); i++) {
            weights.at(i) = std::cos(i * 3.14 / 6);
        }
        auto weightsFP32 = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, weightsShape, weights.data());

        const size_t weightsLevels = 255;

        const auto weightsInLow = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1}, {0.0f}, false);
        const auto weightsInHigh = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1}, {255.0f}, false);

        std::vector<float> perChannelLow(weightsShape[0]);
        std::vector<float> perChannelHigh(weightsShape[0]);

        for (size_t i = 0; i < weightsShape[0]; ++i) {
            perChannelLow[i] = 0.0f;
            perChannelHigh[i] = 255.0f;
        }

        const auto weightsOutLow = ngraph::builder::makeConstant<float>(ngraph::element::f32, {weightsShape[0], 1, 1, 1}, perChannelLow, false);
        const auto weightsOutHigh = ngraph::builder::makeConstant<float>(ngraph::element::f32, {weightsShape[0], 1, 1, 1}, perChannelHigh, false);

        const auto weightsFq = std::make_shared<ngraph::opset2::FakeQuantize>(weightsFP32, weightsInLow, weightsInHigh, weightsOutLow, weightsOutHigh, weightsLevels);

        const ngraph::Strides strides = test_params._strides;
        const ngraph::CoordinateDiff pads_begin = test_params._pads_begin;
        const ngraph::CoordinateDiff pads_end = test_params._pads_end;
        const ngraph::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ngraph::opset2::Convolution>(pool, weightsFq, strides, pads_begin, pads_end, dilations);

        const std::vector<float> outLow = {0.0f};
        const std::vector<float> outHigh = {255.0f};
        const auto result = ngraph::builder::makeFakeQuantize(conv, ngraph::element::f32, dataLevels, {}, outLow, outHigh, outLow, outHigh);

        const ngraph::ResultVector results{
            std::make_shared<ngraph::opset1::Result>(result)
        };
        function = std::make_shared<ngraph::Function>(results, params, "KmbAsymmetricStrideConvSubGraphTest");

        threshold = 0.1f;
    }
};

TEST_P(KmbAsymmetricStrideConvSubGraphTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke, KmbAsymmetricStrideConvSubGraphTest,
    ::testing::Values(
    KmbAsymmetricStrideConvSubGraphTestParams {
        LayerTestsUtils::testPlatformTargetDevice,  // _device
        {1, 1, 16, 16},    // in dims
        {2, 1, 1, 2},      // weights dims
        {1, 2},            // strides
        {0, 0},            // pads_begin
        {0, 0},            // pads_end
    },
    KmbAsymmetricStrideConvSubGraphTestParams {
        LayerTestsUtils::testPlatformTargetDevice,  // _device
        {1, 16, 64, 64},   // in dims
        {16, 16, 1, 2},    // weights dims
        {1, 2},            // strides
        {0, 0},            // pads_begin
        {0, 0},            // pads_end
    })
);


}
