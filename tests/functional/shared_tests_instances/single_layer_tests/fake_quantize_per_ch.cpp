// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

typedef std::tuple<LayerTestsUtils::TargetDevice, InferenceEngine::SizeVector> FakeQuantPerChTestParams;

// Test purpose: have a 'FakeQuantize' split into 'Quantize' + 'Dequantize'
// In HW-pipeline, 'Quantize' will run on DPU, 'Dequantize' on Shave
class KmbFakeQuantPerChTest_VPU3720 :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<FakeQuantPerChTestParams> {
    InferenceEngine::Precision iePrc = InferenceEngine::Precision::FP16;

    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(iePrc);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(iePrc);
        cnnNetwork.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
        cnnNetwork.getOutputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
    }

    void SetUp() override {
        InferenceEngine::SizeVector shape;
        std::tie(targetDevice, shape) = GetParam();
        const auto C = shape[1];
        const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(iePrc);

        const auto params = ngraph::builder::makeParams(ngPrc, {shape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const size_t levels = 256;

        // Different low/high limits per channel (must include 0 in range, so it splits into Q/DQ)
        // In 'DefaultHW' pipeline:
        // 'Quantize' will run on DPU
        // 'Dequantize' will run as SW-kernel
        std::vector<float> lo(C);
        std::vector<float> hi(C);
        for (size_t i = 0; i < C; ++i) {
            lo[i] = 0.0f;
            hi[i] = 8.0f + 0.2f * i * (i % 2 ? -1 : +1);
        }

        const auto dataFq =
                ngraph::builder::makeFakeQuantize(paramOuts[0], ngPrc, levels, {1, C, 1, 1}, lo, hi, lo, hi);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(dataFq)};
        function = std::make_shared<ngraph::Function>(results, params, "FakeQuantPerCh");

        threshold = 0.2f;
    }
};

class KmbFakeQuantPerChTest_VPU3720_SW : public KmbFakeQuantPerChTest_VPU3720 {};
class KmbFakeQuantPerChTest_VPU3720_HW : public KmbFakeQuantPerChTest_VPU3720 {};

TEST_P(KmbFakeQuantPerChTest_VPU3720_HW, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(KmbFakeQuantPerChTest_VPU3720_SW, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

const std::vector<InferenceEngine::SizeVector> shapesHW = {
        {1, 16, 8, 32},
        {1, 32, 16, 8},
};

const std::vector<InferenceEngine::SizeVector> shapesSW = {
        {1, 1, 1, 100}, {1, 3, 8, 32}, {1, 8, 3, 21}, {1, 13, 16, 8}, {1, 16, 3, 5}, {1, 21, 2, 3},
};

INSTANTIATE_TEST_CASE_P(smoke_precommit_, KmbFakeQuantPerChTest_VPU3720_HW,
                        ::testing::Combine(::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                           ::testing::ValuesIn(shapesHW)));

INSTANTIATE_TEST_CASE_P(smoke, KmbFakeQuantPerChTest_VPU3720_SW,
                        ::testing::Combine(::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                           ::testing::ValuesIn(shapesSW)));

}  // namespace
