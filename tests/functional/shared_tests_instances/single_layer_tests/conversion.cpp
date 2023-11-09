//
// Copyright (C) 2019-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/conversion.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXConversionLayerTest : public ConversionLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class VPUXConversionLayerTest_VPU3700 : public VPUXConversionLayerTest {};
class VPUXConversionLayerTest_VPU3720 : public VPUXConversionLayerTest {};
using VPUXConversionLayerTest_VPU3720_ELF = VPUXConversionLayerTest_VPU3720;

TEST_P(VPUXConversionLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXConversionLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXConversionLayerTest_VPU3720_ELF, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    useELFCompilerBackend();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::helpers::ConversionTypes> conversionOpTypes = {
        ngraph::helpers::ConversionTypes::CONVERT,
        ngraph::helpers::ConversionTypes::CONVERT_LIKE,
};

const std::vector<std::vector<size_t>> inShape = {{1, 2, 3, 4}};

// Precision I8 was deleted from netPrecisions because it is not supported by run-time and compiler.
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16, InferenceEngine::Precision::U8};

const std::vector<InferenceEngine::Precision> netPrecisions_VPU3720 = {
        InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16, InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I8, InferenceEngine::Precision::I32};

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_NoReshape, VPUXConversionLayerTest_VPU3700,
                         ::testing::Combine(::testing::ValuesIn(conversionOpTypes), ::testing::Values(inShape),
                                            ::testing::ValuesIn(netPrecisions), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         ConversionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Conversion_NoReshape, VPUXConversionLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(ngraph::helpers::ConversionTypes::CONVERT),
                                            ::testing::Values(inShape), ::testing::ValuesIn(netPrecisions_VPU3720),
                                            ::testing::ValuesIn(netPrecisions_VPU3720),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         ConversionLayerTest::getTestCaseName);

// ------ ELF ------

INSTANTIATE_TEST_SUITE_P(NoReshape, VPUXConversionLayerTest_VPU3720_ELF,
                         ::testing::Combine(::testing::Values(ngraph::helpers::ConversionTypes::CONVERT),
                                            ::testing::Values(inShape), ::testing::ValuesIn(netPrecisions_VPU3720),
                                            ::testing::ValuesIn(netPrecisions_VPU3720),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(InferenceEngine::Layout::NHWC),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         ConversionLayerTest::getTestCaseName);

}  // namespace
