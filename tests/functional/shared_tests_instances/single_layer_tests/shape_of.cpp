//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "kmb_layer_test.hpp"
#include "single_layer_tests/shape_of.hpp"

namespace LayerTestsDefinitions {
class VPUXShapeOfLayerTest : public ShapeOfLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXShapeOfLayerTest_VPU3700 : public VPUXShapeOfLayerTest {
    void SkipBeforeLoad() override {
    }
};

class VPUXShapeOfLayerTest_VPU3720 : public VPUXShapeOfLayerTest {};

TEST_P(VPUXShapeOfLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXShapeOfLayerTest_VPU3720, SW) {
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

// All test instances have the same error:
// C++ exception with description "Check 'm_output_type == element::i64 || m_output_type == element::i32'
// failed at core/src/op/shape_of.cpp:48:
// While validating node 'v3::ShapeOf ShapeOf_1 (Parameter_0[0]:f32{10,10,10}) -> (dynamic?)' with
// friendly_name 'ShapeOf_1': Output type must be i32 or i64" thrown in SetUp().
// [Track number: S#49606]
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Check, VPUXShapeOfLayerTest_VPU3700,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::I64),
                                            ::testing::Values(std::vector<size_t>({10, 10, 10})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         ShapeOfLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_ShapeOf, VPUXShapeOfLayerTest_VPU3700,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::I32),
                                            ::testing::ValuesIn(inShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         ShapeOfLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf_VPU3720, VPUXShapeOfLayerTest_VPU3720,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::I32),
                                            ::testing::ValuesIn(inShapes),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         ShapeOfLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ShapeOf_VPU3720, VPUXShapeOfLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP16),
                                            ::testing::Values(InferenceEngine::Precision::I32),
                                            ::testing::ValuesIn(inShapes_precommit),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         ShapeOfLayerTest::getTestCaseName);
}  // namespace
