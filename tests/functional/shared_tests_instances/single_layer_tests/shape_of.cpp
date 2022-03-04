// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/shape_of.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class KmbShapeOfLayerTest: public ShapeOfLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {

    void SkipBeforeLoad() override {
        if (isCompilerMCM()) {
            throw LayerTestsUtils::KmbSkipTestException("ShapeOf not supported for MCM");
        }
    }

    };

    TEST_P(KmbShapeOfLayerTest, CompareWithRefs) {
        Run();
    }

    TEST_P(KmbShapeOfLayerTest, CompareWithRefs_MLIR) {
        useCompilerMLIR();
        Run();
    }
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP16,
            InferenceEngine::Precision::U8
    };

    const std::vector<std::vector<size_t>> inShapes = {
          std::vector<size_t>{10},
          std::vector<size_t>{10, 11},
          std::vector<size_t>{10, 11, 12},
          std::vector<size_t>{10, 11, 12, 13},
          std::vector<size_t>{10, 11, 12, 13, 14},
          std::vector<size_t>{2, 3, 244, 244},
          std::vector<size_t>{2, 4, 8, 16, 32},
    };

    // All test instances have the same error:
    // C++ exception with description "Check 'm_output_type == element::i64 || m_output_type == element::i32'
    // failed at core/src/op/shape_of.cpp:48:
    // While validating node 'v3::ShapeOf ShapeOf_1 (Parameter_0[0]:f32{10,10,10}) -> (dynamic?)' with
    // friendly_name 'ShapeOf_1': Output type must be i32 or i64" thrown in SetUp().
    // [Track number: S#49606]
    INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Check, KmbShapeOfLayerTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::I64),
                                    ::testing::Values(std::vector<size_t>({10, 10, 10})),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            ShapeOfLayerTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf, KmbShapeOfLayerTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(InferenceEngine::Precision::I32),
                                    ::testing::ValuesIn(inShapes),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                            ShapeOfLayerTest::getTestCaseName);
}  // namespace
