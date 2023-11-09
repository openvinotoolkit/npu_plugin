//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpu_ov1_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class VPUXConvClampSubGraphTest_VPU3700 :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
        public testing::WithParamInterface<LayerTestsUtils::TargetDevice> {
    void SetUp() override {
        const InferenceEngine::SizeVector inputShape{1, 3, 62, 62};
        const InferenceEngine::SizeVector weightsShape{48, 3, 3, 3};

        const auto weights = ngraph::builder::makeConstant<uint8_t>(ngraph::element::f32, weightsShape, {2}, false);
        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const ngraph::Strides strides = {1, 1};
        const ngraph::CoordinateDiff pads_begin = {0, 0};
        const ngraph::CoordinateDiff pads_end = {0, 0};
        const ngraph::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ngraph::opset2::Convolution>(paramOuts[0], weights, strides, pads_begin,
                                                                        pads_end, dilations);

        const auto clamp = std::make_shared<ngraph::op::v0::Clamp>(conv, -1.0f, 1.0f);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(clamp)};
        function = std::make_shared<ngraph::Function>(results, params, "VPUXConvClamp");

        targetDevice = GetParam();
        threshold = 0.1f;
    }
};

TEST_P(VPUXConvClampSubGraphTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke_ConvClamp, VPUXConvClampSubGraphTest_VPU3700,
                        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

}  // namespace
