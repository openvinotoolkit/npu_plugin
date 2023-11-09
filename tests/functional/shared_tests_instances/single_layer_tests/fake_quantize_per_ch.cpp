//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpu_ov1_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include "vpux_private_config.hpp"

namespace {

typedef std::tuple<LayerTestsUtils::TargetDevice, InferenceEngine::SizeVector> FakeQuantPerChTestParams;

// Test purpose: have a 'FakeQuantize' split into 'Quantize' + 'Dequantize'
// In HW-pipeline, 'Quantize' will run on DPU, 'Dequantize' on Shave
class VPUXFakeQuantPerChTest_VPU3720 :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
        public testing::WithParamInterface<FakeQuantPerChTestParams> {
    InferenceEngine::Precision iePrc = InferenceEngine::Precision::FP16;

    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(iePrc);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(iePrc);
        cnnNetwork.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NCHW);
        cnnNetwork.getOutputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NCHW);
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
            if (hi[i] < lo[i]) {
                hi[i] = 8.0f + 0.2f * i;
            }
        }

        const auto dataFq =
                ngraph::builder::makeFakeQuantize(paramOuts[0], ngPrc, levels, {1, C, 1, 1}, lo, hi, lo, hi);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(dataFq)};
        function = std::make_shared<ngraph::Function>(results, params, "FakeQuantPerCh");

        threshold = 0.2f;
    }
};

class VPUXFakeQuantPerChTest_VPU3720_SW : public VPUXFakeQuantPerChTest_VPU3720 {
    void SkipBeforeLoad() override {
        configuration[VPUX_CONFIG_KEY(COMPILATION_MODE_PARAMS)] = "merge-fake-quant=false";
    }
};
class VPUXFakeQuantPerChTest_VPU3720_HW : public VPUXFakeQuantPerChTest_VPU3720 {};
typedef std::tuple<LayerTestsUtils::TargetDevice, InferenceEngine::SizeVector, std::vector<float>, std::vector<float>,
                   std::vector<float>, std::vector<float>>
        FakeQuantPerChCustomLimitsTestParams;

// Test purpose: checking the functional results of the FQ Operation executed on shave for different ZPs for both
// input and output
class VPUXFakeQuantPerChCustomLimitsTest :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
        public testing::WithParamInterface<FakeQuantPerChCustomLimitsTestParams> {
    InferenceEngine::Precision iePrc = InferenceEngine::Precision::FP16;

    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(iePrc);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(iePrc);
        cnnNetwork.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NCHW);
        cnnNetwork.getOutputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NCHW);
    }

    void SetUp() override {
        InferenceEngine::SizeVector shape;
        std::vector<float> ho;
        std::vector<float> lo;
        std::vector<float> hi;
        std::vector<float> li;
        std::tie(targetDevice, shape, ho, lo, hi, li) = this->GetParam();
        const auto C = shape[1];
        const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(iePrc);

        const auto params = ngraph::builder::makeParams(ngPrc, {shape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const size_t levels = 256;

        const auto dataFq =
                ngraph::builder::makeFakeQuantize(paramOuts[0], ngPrc, levels, {1, C, 1, 1}, li, hi, lo, ho);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(dataFq)};
        function = std::make_shared<ngraph::Function>(results, params, "FakeQuantPerCh");
    }
};

class VPUXFakeQuantPerChCustomLimitsTest_VPU3720_SW : public VPUXFakeQuantPerChCustomLimitsTest {};

TEST_P(VPUXFakeQuantPerChTest_VPU3720_HW, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXFakeQuantPerChTest_VPU3720_SW, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXFakeQuantPerChCustomLimitsTest_VPU3720_SW, SW) {
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

const std::vector<InferenceEngine::SizeVector> shapesSWcustomLimits = {{1, 3, 199, 199}};

const std::vector<InferenceEngine::SizeVector> shapesTiling = {
        {1, 64, 128, 100}, {1, 128, 68, 164},  // aclnet
};

INSTANTIATE_TEST_CASE_P(smoke_precommit_FakeQuantPerCh, VPUXFakeQuantPerChTest_VPU3720_HW,
                        ::testing::Combine(::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                           ::testing::ValuesIn(shapesHW)));

INSTANTIATE_TEST_CASE_P(smoke_FakeQuantPerCh, VPUXFakeQuantPerChTest_VPU3720_SW,
                        ::testing::Combine(::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                           ::testing::ValuesIn(shapesSW)));

INSTANTIATE_TEST_CASE_P(smoke_tiling_FakeQuantPerCh, VPUXFakeQuantPerChTest_VPU3720_SW,
                        ::testing::Combine(::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                           ::testing::ValuesIn(shapesTiling)));

//{outHigh, outLow, inHigh, inLow}
// testing per-channel quantization with different ZPs for output
INSTANTIATE_TEST_CASE_P(
        smoke_customLimits_FakeQuantPerCh1, VPUXFakeQuantPerChCustomLimitsTest_VPU3720_SW,
        ::testing::Combine(::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                           ::testing::ValuesIn(shapesSWcustomLimits),
                           ::testing::Values(std::vector<float>{+2.63867188}),
                           ::testing::Values(std::vector<float>{-49.28125, -35.65625, -31.828125}),
                           ::testing::Values(std::vector<float>{+2.551250e+02, +2.670000e+02, +2.780000e+02}),
                           ::testing::Values(std::vector<float>{+2.551250e+02, +2.670000e+02, +2.780000e+02})));

// testing per-channel quantization with different ZPs for output and input
INSTANTIATE_TEST_CASE_P(
        smoke_customLimits_FakeQuantPerCh2, VPUXFakeQuantPerChCustomLimitsTest_VPU3720_SW,
        ::testing::Combine(::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                           ::testing::ValuesIn(shapesSWcustomLimits),
                           ::testing::Values(std::vector<float>{+2.551250e+02, +2.670000e+02, +2.780000e+02}),
                           ::testing::Values(std::vector<float>{-49.28125, -35.65625, -31.828125}),
                           ::testing::Values(std::vector<float>{+2.551250e+02, +2.670000e+02, +2.780000e+02}),
                           ::testing::Values(std::vector<float>{-49.28125, -35.65625, -31.828125})));

// testing per-channel quantization with different ZPs for input
INSTANTIATE_TEST_CASE_P(smoke_customLimits_FakeQuantPerCh3, VPUXFakeQuantPerChCustomLimitsTest_VPU3720_SW,
                        ::testing::Combine(::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                           ::testing::ValuesIn(shapesSWcustomLimits),
                                           ::testing::Values(std::vector<float>{+2.63867188}),
                                           ::testing::Values(std::vector<float>{-2.63867188}),
                                           ::testing::Values(std::vector<float>{+2.551250e+02, +2.670000e+02,
                                                                                +2.780000e+02}),
                                           ::testing::Values(std::vector<float>{-49.28125, -35.65625, -31.828125})));

}  // namespace
