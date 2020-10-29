// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/power.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbPowerLayerTest : public PowerLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbPowerLayerTest, PowerCheck) {
    Run();
}
} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

std::vector<std::vector<std::vector<size_t>>> inShapes = {
    {{1, 8}},
    {{2, 16}},
    {{3, 32}},
    {{4, 64}},
    {{5, 128}},
    {{6, 256}},
    {{7, 512}},
    {{8, 1024}}
};

std::vector<std::vector<float >> Power = {
    {0.0f},
    {0.5f},
    {1.0f},
    {1.1f},
    {1.5f},
    {2.0f},
};

std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                         InferenceEngine::Precision::FP16,
};

// Test works only when power = 1, in all other cases there are similar errors:
// C++ exception with description "Operation PowerIE Power_xxxxx has unsupported power N (where N unequals to 1)
// kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:640
// openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
// [Track number: S#41811]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_power, KmbPowerLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(inShapes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::ValuesIn(Power)),
                        KmbPowerLayerTest::getTestCaseName);

// Subset of parameters and additional test for Power layer.
// This subset is used to enable test on Power layer.
// Do not forget to remove this subset and test when initial test DISABLED_smoke_power will be enabled.
std::vector<std::vector<float >> Power_pass_mcm = {
    {1.0f},
};

INSTANTIATE_TEST_CASE_P(smoke_power_pass_mcm, KmbPowerLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(inShapes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::ValuesIn(Power_pass_mcm)),
                        KmbPowerLayerTest::getTestCaseName);
// End of additional test and its parameters

}  // namespace
