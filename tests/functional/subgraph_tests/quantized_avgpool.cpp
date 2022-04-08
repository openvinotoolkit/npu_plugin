//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class KmbQuantizedAvgPoolSubGraphTest :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<LayerTestsUtils::TargetDevice> {
    void SetUp() override {
        const InferenceEngine::SizeVector inputShape{1, 16, 32, 32};

        const auto params = ngraph::builder::makeParams(ngraph::element::f16, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const size_t dataLevels = 256;
        const std::vector<float> dataLow = {0.0f};
        const std::vector<float> dataHigh = {100.0f};
        const auto dataFq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngraph::element::f32, dataLevels, {},
                                                              dataLow, dataHigh, dataLow, dataHigh);

        const ngraph::Strides strides        = {2, 2};
        const std::vector<size_t> pads_begin = {0, 0};
        const std::vector<size_t> pads_end   = {0, 0};
        const ngraph::Strides dilations      = {1, 1};
        const std::vector<size_t> kernelSize = {2, 2};
        const ngraph::op::PadType padType = ngraph::op::PadType::AUTO;
        const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;

        const auto pooling = ngraph::builder::makePooling(dataFq, strides, pads_begin, pads_end,
                kernelSize, roundingType, padType, false, ngraph::helpers::PoolingTypes::AVG);

        const std::vector<float> outDataLow = {0.0f};
        const std::vector<float> outDataHigh = {100.0f};
        const auto outDataFq = ngraph::builder::makeFakeQuantize(pooling, ngraph::element::f32, dataLevels, {},
                                                                 outDataLow, outDataHigh, outDataLow, outDataHigh);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(outDataFq)};
        function = std::make_shared<ngraph::Function>(results, params, "KmbQuantizedAvgPool");

        targetDevice = GetParam();
        threshold = 0.5f;
    }
};

TEST_P(KmbQuantizedAvgPoolSubGraphTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbQuantizedAvgPoolSubGraphTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke, KmbQuantizedAvgPoolSubGraphTest,
                        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

}  // namespace
