//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/lrn.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbLrnLayerTest : public LrnLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

class KmbLrnLayerTest_VPU3720 : public KmbLrnLayerTest {};

TEST_P(KmbLrnLayerTest, LrnCheck) {
    Run();
}

TEST_P(KmbLrnLayerTest, LrnCheck_MLIR) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbLrnLayerTest_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
// Common params

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<std::vector<int64_t>> axes = {{1}, {2, 3}};

const double alpha = 9.9e-05;
const double beta = 2;
const double bias = 1.0;
const size_t size = 5;

INSTANTIATE_TEST_SUITE_P(smoke_LrnCheck, KmbLrnLayerTest,
                         ::testing::Combine(::testing::Values(alpha), ::testing::Values(beta), ::testing::Values(bias),
                                            ::testing::Values(size), ::testing::ValuesIn(axes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(std::vector<size_t>({1, 10, 3, 2})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbLrnLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> axes_vpu3720 = {{2}, {1, 2}};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_LrnCheck, KmbLrnLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(alpha), ::testing::Values(beta), ::testing::Values(bias),
                                            ::testing::Values(size), ::testing::ValuesIn(axes_vpu3720),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(std::vector<size_t>({1, 10, 3, 2})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         KmbLrnLayerTest::getTestCaseName);

}  // namespace
