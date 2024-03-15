//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/lrn.hpp"
#include <vector>
#include "vpu_ov1_layer_test.hpp"
#include "vpux_private_properties.hpp"

namespace LayerTestsDefinitions {

class LrnLayerTestCommon : public LrnLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class LrnLayerTest_NPU3700 : public LrnLayerTestCommon {};
class LrnLayerTest_SW_FP16 : public LrnLayerTestCommon {};
class LrnLayerTest_HW_FP16 : public LrnLayerTestCommon {};
class LrnLayerTest_SW_FP32 : public LrnLayerTestCommon {
    void ConfigureNetwork() override {
        configuration[ov::intel_vpux::compilation_mode_params.name()] = "convert-precision-to-fp16=false";
    }
};

TEST_P(LrnLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

// FP16
TEST_P(LrnLayerTest_HW_FP16, NPU3720) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(LrnLayerTest_SW_FP32, NPU3720) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const double alpha = 9.9e-05;
const double beta = 2;
const double bias = 1.0;
const size_t size = 5;

//
// NPU3700
//
const std::vector<std::vector<int64_t>> axes_3700 = {{1}, {2, 3}};

INSTANTIATE_TEST_SUITE_P(smoke_LrnCheck, LrnLayerTest_NPU3700,
                         ::testing::Combine(::testing::Values(alpha), ::testing::Values(beta), ::testing::Values(bias),
                                            ::testing::Values(size), ::testing::ValuesIn(axes_3700),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(std::vector<size_t>({1, 10, 3, 2})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         LrnLayerTest_NPU3700::getTestCaseName);

//
// NPU3720
//
const std::vector<std::vector<int64_t>> axes = {{1}, {2}, {1, 2}, {2, 3}, {1, 2, 3}};

const auto lrnParams_FP16 =
        ::testing::Combine(::testing::Values(alpha), ::testing::Values(beta), ::testing::Values(bias),
                           ::testing::Values(size), ::testing::ValuesIn(axes), ::testing::ValuesIn(netPrecisions),
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                           ::testing::Values(std::vector<size_t>({1, 10, 3, 2})),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto lrnGooglenetV1Params_FP16 = ::testing::Combine(
        ::testing::Values(9.9e-05),                    // alpha
        ::testing::Values(0.75),                       // beta
        ::testing::Values(1.0),                        // bias
        ::testing::Values(5),                          // size
        ::testing::Values(std::vector<int64_t>({1})),  // axes
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(std::vector<size_t>({1, 64, 56, 56})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto lrnParams_FP32 = ::testing::Combine(
        ::testing::Values(9.9e-05),                    // alpha
        ::testing::Values(0.75),                       // beta
        ::testing::Values(1.0),                        // bias
        ::testing::Values(5),                          // size
        ::testing::Values(std::vector<int64_t>({1})),  // axes
        ::testing::Values(InferenceEngine::Precision::FP32), ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32), ::testing::Values(std::vector<size_t>({1, 32, 56, 56})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

//
// NPU3720 Instantiation
// FP16 HW

INSTANTIATE_TEST_SUITE_P(smoke_precommit_LrnCheck, LrnLayerTest_HW_FP16, lrnParams_FP16,
                         LrnLayerTest_HW_FP16::getTestCaseName);

//
// NPU3720 Instantiation
// FP32 SW

INSTANTIATE_TEST_SUITE_P(smoke_precommit_LrnGooglenetV1_FP32, LrnLayerTest_SW_FP32, lrnParams_FP32,
                         LrnLayerTest_SW_FP32::getTestCaseName);

}  // namespace
