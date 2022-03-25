//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/pad.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class KmbPadLayerTest_MLIR_ONLY: public PadLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
    class KmbPadLayerTest: public PadLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

    TEST_P(KmbPadLayerTest_MLIR_ONLY, CompareWithRefs_MLIR_ONLY) {
        useCompilerMLIR();
        Run();
    }

    TEST_P(KmbPadLayerTest, CompareWithRefs_MLIR) {
        useCompilerMLIR();
        Run();
    }

    TEST_P(KmbPadLayerTest, CompareWithRefs) {
        Run();
    }
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP16
    };

    const std::vector<std::vector<int64_t>> padsBegin4D = {{0, 0, 0, 0}, {0, 1, 1, 1}, {0, 0, 1, 0}, {0, 3, 0, 1}};
    const std::vector<std::vector<int64_t>> padsEnd4D   = {{0, 0, 0, 0}, {0, 1, 1, 1}, {0, 0, 0, 1}, {0, 3, 2, 0}};

    const std::vector<float> argPadValue = {0.f, 1.f, 2.f, -1.f};

    const std::vector<ngraph::helpers::PadMode> padMode = {
            ngraph::helpers::PadMode::EDGE,
            ngraph::helpers::PadMode::REFLECT,
            ngraph::helpers::PadMode::SYMMETRIC
    };

    const auto pad4DConstparams = testing::Combine(
            testing::ValuesIn(padsBegin4D),
            testing::ValuesIn(padsEnd4D),
            testing::ValuesIn(argPadValue),
            testing::Values(ngraph::helpers::PadMode::CONSTANT),
            testing::ValuesIn(netPrecisions),
            testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Layout::NCHW),
            testing::Values(std::vector<size_t>{1, 5, 10, 11}),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    // [Track number: E#13236]
    INSTANTIATE_TEST_SUITE_P(
            DISABLED_smoke_Pad4DConst,
            KmbPadLayerTest,
            pad4DConstparams,
            KmbPadLayerTest::getTestCaseName
    );

    const auto pad4Dparams = testing::Combine(
            testing::ValuesIn(padsBegin4D),
            testing::ValuesIn(padsEnd4D),
            testing::Values(0),
            testing::ValuesIn(padMode),
            testing::ValuesIn(netPrecisions),
            testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Layout::NCHW),
            testing::Values(std::vector<size_t>{1, 5, 10, 11}),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    INSTANTIATE_TEST_SUITE_P(
            smoke_Pad4D,
            KmbPadLayerTest,
            pad4Dparams,
            KmbPadLayerTest::getTestCaseName
    );

    const std::vector<std::vector<int64_t>> padsBeginForConcat = {{0, 0, 0, 0}, {4, 2, 1, 3}, {8, 0, 0, 0}, {0, 0, 2, 0}};
    const std::vector<std::vector<int64_t>> padsEndForConcat   = {{0, 0, 0, 0}, {5, 2, 6, 1}, {8, 0, 0, 0}, {0, 1, 0, 3}};

    const auto padConvertToConcat = testing::Combine(
            testing::ValuesIn(padsBeginForConcat),
            testing::ValuesIn(padsEndForConcat),
            testing::Values(0, 1),
            testing::Values(ngraph::helpers::PadMode::CONSTANT),
            testing::ValuesIn(netPrecisions),
            testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
            testing::Values(std::vector<size_t>{1, 10, 20, 30}),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    INSTANTIATE_TEST_SUITE_P(
            smoke_PadConvertToConcat,
            KmbPadLayerTest_MLIR_ONLY,
            padConvertToConcat,
            KmbPadLayerTest::getTestCaseName
    );

}  // namespace
