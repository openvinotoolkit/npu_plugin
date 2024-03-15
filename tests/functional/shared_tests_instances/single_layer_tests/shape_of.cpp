//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "single_layer_tests/shape_of.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class ShapeOfLayerTestCommon : public ShapeOfLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class ShapeOfLayerTest_NPU3700 : public ShapeOfLayerTestCommon {};
class ShapeOfLayerTest_NPU3720 : public ShapeOfLayerTestCommon {};

TEST_P(ShapeOfLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(ShapeOfLayerTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16,
                                                               InferenceEngine::Precision::U8};

const std::vector<std::vector<size_t>> inShapes = {
        std::vector<size_t>{10},
        std::vector<size_t>{10, 11},
        std::vector<size_t>{10, 11, 12},
        std::vector<size_t>{10, 11, 12, 13},
        std::vector<size_t>{10, 11, 12, 13, 14},
        std::vector<size_t>{2, 3, 244, 244},
        std::vector<size_t>{2, 4, 8, 16, 32},
};

const std::vector<std::vector<size_t>> inShapes_precommit = {
        std::vector<size_t>{3, 3, 5},
        std::vector<size_t>{5, 7, 6, 3},
};

const auto paramsConfig1 =
        testing::Combine(::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::I32),
                         ::testing::ValuesIn(inShapes), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));
const auto paramsPrecommit = testing::Combine(
        ::testing::Values(InferenceEngine::Precision::FP16), ::testing::Values(InferenceEngine::Precision::I32),
        ::testing::ValuesIn(inShapes_precommit), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// --------- NPU3700 ---------
// Tracking number [E#85137]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Check, ShapeOfLayerTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::I64),
                                            ::testing::Values(std::vector<size_t>({10, 10, 10})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         ShapeOfLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf, ShapeOfLayerTest_NPU3700, paramsConfig1, ShapeOfLayerTest::getTestCaseName);

// --------- NPU3720 ---------

INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf, ShapeOfLayerTest_NPU3720, paramsConfig1, ShapeOfLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ShapeOf, ShapeOfLayerTest_NPU3720, paramsPrecommit,
                         ShapeOfLayerTest::getTestCaseName);

}  // namespace
