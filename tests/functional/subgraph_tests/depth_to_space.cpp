//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common/functions.h"
#include "vpu_ov1_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
/*Creates a graph :
Conv         Input
\             /
D2S          /
  \        /
 Eltiwise
    |
 Result
*/
namespace {

class VPUXConvConcatReshape_D2S_VPU3720 :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
        public testing::WithParamInterface<LayerTestsUtils::TargetDevice> {
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice();
        inPrc = InferenceEngine::Precision::FP16;
        outPrc = InferenceEngine::Precision::FP16;
        const InferenceEngine::SizeVector input_1Shape{1, 3, 512, 512};
        const InferenceEngine::SizeVector input_2Shape{1, 32, 256, 256};
        const InferenceEngine::SizeVector weightsShape{12, 32, 1, 1};
        auto params = ngraph::builder::makeParams(ngraph::element::f16, {input_1Shape, input_2Shape});
        const auto weights = ngraph::builder::makeConstant<float>(ngraph::element::f16, weightsShape, {0.0f}, false);
        const ngraph::Strides strides = {1, 1};
        const ngraph::CoordinateDiff pads_begin = {0, 0};
        const ngraph::CoordinateDiff pads_end = {0, 0};
        const ngraph::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ngraph::opset2::Convolution>(params[1], weights, strides, pads_begin,
                                                                        pads_end, dilations);
        const auto D2S = std::make_shared<ngraph::opset1::DepthToSpace>(conv, "blocks_first", 2UL);
        const auto addOp = std::make_shared<ngraph::opset1::Add>(D2S, params[0]);
        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(addOp)};
        function = std::make_shared<ngraph::Function>(results, params, "VPUXConvConcatReshape_D2S_VPU3720");
        threshold = 1.0f;
    }
};

TEST_P(VPUXConvConcatReshape_D2S_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}
INSTANTIATE_TEST_SUITE_P(smoke_precommit_depth_to_space_test, VPUXConvConcatReshape_D2S_VPU3720,
                         ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
}  // namespace
