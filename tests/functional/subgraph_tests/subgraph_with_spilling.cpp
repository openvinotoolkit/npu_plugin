// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

struct ScheduleSubGraphSpillingTestParams {
    LayerTestsUtils::TargetDevice _device;
    InferenceEngine::SizeVector _in_dims;
    InferenceEngine::SizeVector _w_dims_conv_1;
    InferenceEngine::SizeVector _w_dims_conv_2;
    std::vector<uint64_t> _strides;
    std::vector<int64_t> _pads_begin;
    std::vector<int64_t> _pads_end;
};

// Create a simple network of DPU tasks that when scheduled will cause spilling
//                  |-> Conv1 -> Conv2 -> |
// Input -> MaxPool |                     | Eltwise -> Output
//                  |-------------------> |
class VPUXScheduleSubGraphSpillingTest_VPU3700 :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<ScheduleSubGraphSpillingTestParams> {
    void SetUp() override {
        const auto test_params = GetParam();
        targetDevice = test_params._device;
        const InferenceEngine::SizeVector inputShape = test_params._in_dims;
        const InferenceEngine::SizeVector weights1Shape = test_params._w_dims_conv_1;
        const InferenceEngine::SizeVector weights2Shape = test_params._w_dims_conv_2;

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        std::vector<uint64_t> poolStridesVec = {1, 1};
        std::vector<uint64_t> poolKernelVec = {1, 1};
        const ngraph::Strides poolStrides = poolStridesVec;
        const ngraph::Shape padsBegin = {0, 0};
        const ngraph::Shape padsEnd = {0, 0};
        const ngraph::Shape poolKernel = poolKernelVec;
        const auto pool =
                std::make_shared<ngraph::opset2::MaxPool>(paramOuts[0], poolStrides, padsBegin, padsEnd, poolKernel);

        const ngraph::Strides strides = test_params._strides;
        const ngraph::CoordinateDiff pads_begin = test_params._pads_begin;
        const ngraph::CoordinateDiff pads_end = test_params._pads_end;
        const ngraph::Strides dilations = {1, 1};

        std::vector<float> weights1(weights1Shape[0] * weights1Shape[1] * weights1Shape[2] * weights1Shape[3], 1);
        auto weights1FP32 =
                std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::f32, weights1Shape, weights1.data());

        const auto conv1 = std::make_shared<ngraph::opset2::Convolution>(pool, weights1FP32, strides, pads_begin,
                                                                         pads_end, dilations);

        std::vector<float> weights2(weights2Shape[0] * weights2Shape[1] * weights2Shape[2] * weights2Shape[3], 1);
        auto weights2FP32 =
                std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::f32, weights2Shape, weights2.data());

        const auto conv2 = std::make_shared<ngraph::opset2::Convolution>(conv1, weights2FP32, strides, pads_begin,
                                                                         pads_end, dilations);

        const auto eltwise = ngraph::builder::makeEltwise(pool, conv2, ngraph::helpers::EltwiseTypes::ADD);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwise)};
        function = std::make_shared<ngraph::Function>(results, params, "VPUXScheduleSubGraphSpillingTest");

        threshold = 0.1f;
    }
};

TEST_P(VPUXScheduleSubGraphSpillingTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke_ScheduleSubGraphSpilling, VPUXScheduleSubGraphSpillingTest_VPU3700,
                        ::testing::Values(ScheduleSubGraphSpillingTestParams{
                                LayerTestsUtils::testPlatformTargetDevice,  // _device
                                {1, 16, 80, 80},                            // in dims
                                {32, 16, 1, 1},                             // weights 1 dims
                                {16, 32, 1, 1},                             // weights 2 dims
                                {1, 1},                                     // strides
                                {0, 0},                                     // pads_begin
                                {0, 0},                                     // pads_end
                        }));

}  // namespace
