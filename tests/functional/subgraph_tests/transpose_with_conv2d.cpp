// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class KmbTransposeWithConv2dTest :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<std::tuple<std::vector<int64_t>>> {
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice;
        const InferenceEngine::SizeVector inputShape = {1, 16, 32, 64};

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const std::vector<int64_t> transposeArg0Weights = std::get<0>(GetParam());
        auto transposeArg0Const = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::i64, ngraph::Shape{transposeArg0Weights.size()}, transposeArg0Weights.data());

        auto transposeArg0Node =
                std::make_shared<ngraph::op::v1::Transpose>(paramOuts.at(0), transposeArg0Const->output(0));

        const size_t planesOut = 16;
        const size_t planesIn = transposeArg0Node->get_shape().at(1);
        const size_t kernelY = 3;
        const size_t kernelX = 3;

        std::vector<float> weights(planesIn * planesOut * kernelY * kernelX);
        for (std::size_t i = 0; i < weights.size(); i++) {
            weights.at(i) = std::cos(i * 3.14 / 6.f);
        }
        auto constLayerNode = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, ngraph::Shape{planesOut, planesIn, kernelY, kernelX}, weights.data());

        auto conv2dNode = std::make_shared<ngraph::op::v1::Convolution>(
                transposeArg0Node->output(0), constLayerNode->output(0), ngraph::Strides(std::vector<size_t>{1, 1}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ngraph::Strides(std::vector<size_t>{1, 1}));

        auto reluNode = std::make_shared<ngraph::op::Relu>(conv2dNode->output(0));
        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(reluNode)};

        function = std::make_shared<ngraph::Function>(results, params, "KmbTransposeWithConv2dTest");

        threshold = 0.5f;
    }
};

TEST_P(KmbTransposeWithConv2dTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbTransposeWithConv2dTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

const std::vector<std::vector<int64_t>> transposes = {
        {0, 3, 1, 2}, {0, 3, 2, 1}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 1, 3, 2},
};

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_conv2d_with_act, KmbTransposeWithConv2dTest,
                         ::testing::Combine(::testing::ValuesIn(transposes)));

}  // namespace
