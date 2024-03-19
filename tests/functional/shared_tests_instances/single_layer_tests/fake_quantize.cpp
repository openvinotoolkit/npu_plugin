//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/fake_quantize.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class FakeQuantizeLayerTestCommon :
        public FakeQuantizeLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class FakeQuantizeLayerTest_SW_NPU3700 : public FakeQuantizeLayerTestCommon {};
class FakeQuantizeLayerTest_HW_NPU3700 : public FakeQuantizeLayerTestCommon {};

class FakeQuantizeLayerTest_SW_NPU3720 : public FakeQuantizeLayerTestCommon {
    // Use realistic float inputs (default generator produces int data)
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        const auto& specificParams = std::get<0>(GetParam());
        const auto& limits = std::get<2>(specificParams);
        float low, high;

        if (limits.empty()) {
            low = 0;  // match 'makeFakeQuantize' default ranges
            high = 12;
        } else {
            low = limits[0];  // use user ranges
            high = limits[1];
        }

        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();
        uint16_t* u16Ptr = blob->buffer().as<uint16_t*>();

        std::mt19937 gen(123);
        const float extra = 0.2f;
        std::uniform_real_distribution<float> dist(low - extra, high + extra);
        for (size_t i = 0; i < blob->size(); i++) {
            float f32 = dist(gen);
            u16Ptr[i] = ngraph::float16(f32).to_bits();
        }
        return blob;
    }
};

class FakeQuantizeLayerTest_HW_NPU3720 : public FakeQuantizeLayerTestCommon {};

// ------ NPU3700 ------
TEST_P(FakeQuantizeLayerTest_SW_NPU3700, SW) {
    setPlatformVPU3700();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(FakeQuantizeLayerTest_HW_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

// ------ NPU3720 ------
TEST_P(FakeQuantizeLayerTest_SW_NPU3720, SW) {
    const auto tol = 1.6;               // To cope with cpu/npu 'limits' diffs
    threshold = fabs(threshold) * tol;  // E#77437
    abs_threshold = threshold;          // Rely on absolute value check
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(FakeQuantizeLayerTest_HW_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<std::vector<size_t>> inputShapes = {{1, 3, 10, 10}};
const std::vector<std::vector<size_t>> constShapes = {{1}, {1, 3, 1, 1}};
const std::vector<size_t> levels = {16, 255, 256};

const std::vector<std::vector<size_t>> inputShapesND = {{1, 512}};
const std::vector<std::vector<size_t>> constShapesND = {{1}};

const std::pair<std::string, std::map<std::string, std::string>> config = {};
const std::vector<float> fqArgs = {0, 255, 0, 255};
const std::vector<float> inputParams = {0, 255, 1};

const auto fqParams =
        ::testing::Combine(::testing::ValuesIn(levels), ::testing::ValuesIn(constShapes), ::testing::Values(fqArgs),
                           ::testing::Values(inputParams), ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY));

const auto fqParamsND =
        ::testing::Combine(::testing::ValuesIn(levels), ::testing::ValuesIn(constShapesND), ::testing::Values(fqArgs),
                           ::testing::Values(inputParams), ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY));

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize, FakeQuantizeLayerTest_SW_NPU3700,
                         ::testing::Combine(fqParams, ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(config)),
                         FakeQuantizeLayerTest_SW_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_ND, FakeQuantizeLayerTest_SW_NPU3700,
                         ::testing::Combine(fqParamsND, ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapesND),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(config)),
                         FakeQuantizeLayerTest_SW_NPU3700::getTestCaseName);

// TODO: support levels=16
// "Can't convert 12 Bit to Byte" while working u4 precision (!quant.uniform<u4:f16, 0.5:128>)
const std::vector<size_t> hw_levels = {255, 256};
const auto hw_fqParams =
        ::testing::Combine(::testing::ValuesIn(hw_levels), ::testing::ValuesIn(constShapes), ::testing::Values(fqArgs),
                           ::testing::Values(inputParams), ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY));

const auto hw_fqParamsND = ::testing::Combine(::testing::ValuesIn(hw_levels), ::testing::ValuesIn(constShapesND),
                                              ::testing::Values(fqArgs), ::testing::Values(inputParams),
                                              ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY));

/* ================================= NPU3700 ================================= */

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize, FakeQuantizeLayerTest_HW_NPU3700,
                         ::testing::Combine(hw_fqParams, ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(config)),
                         FakeQuantizeLayerTest_HW_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_ND, FakeQuantizeLayerTest_HW_NPU3700,
                         ::testing::Combine(hw_fqParamsND, ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapesND),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(config)),
                         FakeQuantizeLayerTest_HW_NPU3700::getTestCaseName);

/* ================================= NPU3720 ================================= */
// Per-Tensor
const std::vector<size_t> u8qLevels = {256};

const std::vector<std::vector<size_t>> inShapes3720 = {
        {2, 3, 10, 10},
        {1, 32, 16, 8},
};

const std::vector<std::vector<size_t>> tilingShapes3720 = {
        {1, 128, 64, 64},
        {1, 64, 128, 80},
        {1, 256, 80, 80},
};

// {inLow, inHigh, outLow, outHigh}
const std::vector<std::vector<float>> fqLimits = {{+0.00, +0.90, +0.00, +0.90}, {+4.50, +9.80, +4.55, +9.74},
                                                  {-5.20, -1.50, -5.15, -1.53}, {-0.50, +0.60, +0.62, -0.58},
                                                  {-0.50, +1.60, -0.40, +1.62}, {-39.0, +231.0, -28.0, +250.0}};

const std::vector<float> inParams = {};  // n/a, overriding GenerateInput
const auto fqParamsU = ::testing::Combine(::testing::ValuesIn(u8qLevels), ::testing::Values(constShapes[0]),
                                          ::testing::ValuesIn(fqLimits), ::testing::Values(inParams),
                                          ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY));

const auto perTensorCfg = ::testing::Combine(
        fqParamsU, ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::ANY), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(inShapes3720), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
        ::testing::Values(config));

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_PerTensor, FakeQuantizeLayerTest_SW_NPU3720, perTensorCfg,
                         FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

// NPU3720 Per-Tensor Tiling
const auto fqParamsT = ::testing::Combine(::testing::ValuesIn(u8qLevels), ::testing::Values(constShapes[0]),
                                          ::testing::Values(fqLimits[0]), ::testing::Values(inParams),
                                          ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY));

INSTANTIATE_TEST_SUITE_P(smoke_tiling_FakeQuantize_PerTensor, FakeQuantizeLayerTest_SW_NPU3720,
                         ::testing::Combine(fqParamsT, ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(tilingShapes3720[2]),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(config)),
                         FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

// NPU3720 Per-Channel (different lo/hi limits per channel)

// Helper to keep 'input' and 'limits' shapes aligned
const auto perChParams(std::vector<size_t> inShape) {
    const auto levels = 255;
    const std::vector<float> noLimits = {};  // empty => per channel default inits
    std::vector<size_t> ctShape = {1, inShape[1], 1, 1};

    const auto fqParams =
            ::testing::Combine(::testing::Values(levels), ::testing::Values(ctShape), ::testing::Values(noLimits),
                               ::testing::Values(inParams), ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY));

    return ::testing::Combine(
            fqParams, ::testing::Values(InferenceEngine::Precision::FP16),
            ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::FP16),
            ::testing::Values(InferenceEngine::Layout::ANY), ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(inShape), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
            ::testing::Values(config));
}

INSTANTIATE_TEST_SUITE_P(smoke_precommit_FakeQuantize_PerCh_a, FakeQuantizeLayerTest_SW_NPU3720,
                         perChParams(std::vector<size_t>{1, 3, 10, 10}),
                         FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_PerCh_b, FakeQuantizeLayerTest_SW_NPU3720,
                         perChParams(std::vector<size_t>{1, 8, 9, 9}),
                         FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_PerCh_c, FakeQuantizeLayerTest_SW_NPU3720,
                         perChParams(std::vector<size_t>{1, 17, 5, 2}),
                         FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_PerCh_d, FakeQuantizeLayerTest_SW_NPU3720,
                         perChParams(std::vector<size_t>{1, 32, 3, 3}),
                         FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

// NPU3720 Per-Channel Tiling tests
INSTANTIATE_TEST_SUITE_P(smoke_tiling_FakeQuantize_PerCh_a, FakeQuantizeLayerTest_SW_NPU3720,
                         perChParams(tilingShapes3720[0]), FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_tiling_FakeQuantize_PerCh_b, FakeQuantizeLayerTest_SW_NPU3720,
                         perChParams(tilingShapes3720[1]), FakeQuantizeLayerTest_SW_NPU3720::getTestCaseName);

// NPU3720 Fp32 input
const std::vector<size_t> levels3720 = {256};
const std::vector<std::vector<size_t>> constShapes3720 = {{1}};
const std::vector<float> fqArgs3720 = {0, 0.631348, 0, 0.631348};
const std::vector<float> inputParams3720 = {0, 255, 1};
const std::vector<InferenceEngine::Precision> netPrecisions3720 = {InferenceEngine::Precision::FP16};
const std::vector<std::vector<size_t>> inputShapes37204d = {{1, 3, 4, 32}, {1, 4, 1, 32}, {1, 1, 4, 32}};
const std::vector<std::vector<size_t>> inputShapes3720nd = {{3, 8, 128}, {4, 8, 128}};

const auto params3720 = ::testing::Combine(::testing::ValuesIn(levels3720), ::testing::ValuesIn(constShapes3720),
                                           ::testing::Values(fqArgs3720), ::testing::Values(inputParams3720),
                                           ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY));

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_ND, FakeQuantizeLayerTest_HW_NPU3720,
                         ::testing::Combine(params3720, ::testing::ValuesIn(netPrecisions3720),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::CHW),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapes3720nd),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(config)),
                         FakeQuantizeLayerTest_HW_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_4D, FakeQuantizeLayerTest_HW_NPU3720,
                         ::testing::Combine(params3720, ::testing::ValuesIn(netPrecisions3720),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Layout::NCHW),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapes37204d),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                            ::testing::Values(config)),
                         FakeQuantizeLayerTest_HW_NPU3720::getTestCaseName);

}  // namespace
