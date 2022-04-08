// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class KmbNormalizeL2FusionSubGraphTest :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<LayerTestsUtils::TargetDevice> {
    void SetUp() override {
        const InferenceEngine::SizeVector inputShape{1, 192};

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const InferenceEngine::SizeVector axesShape{1};
        const auto axes = ngraph::builder::makeConstant<uint8_t>(ngraph::element::u8, axesShape, {}, true, 1, 0);

        auto reduce_l2 = std::make_shared<ngraph::opset4::ReduceL2>(paramOuts[0], axes, true);
        double min, max;
        min = std::numeric_limits<float>::min();
        max = std::numeric_limits<float>::max();
        const auto clamp = std::make_shared<ngraph::op::v0::Clamp>(reduce_l2, min, max);
        auto divide = std::make_shared<ngraph::opset1::Divide>(paramOuts[0], clamp);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(divide)};
        function = std::make_shared<ngraph::Function>(results, params, "KmbNormalizeL2Fusion");

        targetDevice = GetParam();
        threshold = 0.1f;
    }
};

TEST_P(KmbNormalizeL2FusionSubGraphTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke, KmbNormalizeL2FusionSubGraphTest,
                        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

}  // namespace
