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
    class KmbPadLayerTest_MLIR_VPU3720: public KmbPadLayerTest {};

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

    TEST_P(KmbPadLayerTest_MLIR_VPU3720, CompareWithRefs_MLIR_VPU3720) {
        useCompilerMLIR();
        setPlatformVPU3720();
        setDefaultHardwareModeMLIR();
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

    const std::vector<std::vector<int64_t>> padsBegin4D_VPU3720 = {{0, 0, 0, 0}, {0, 3, 0, 1}};
    const std::vector<std::vector<int64_t>> padsEnd4D_VPU3720   = {{0, 0, 0, 0}, {0, 3, 2, 0}};
    const std::vector<float> argPadValue_VPU3720 = {0.f, -1.f};

    const auto pad4DConstparams_VPU3720 = testing::Combine(
            testing::ValuesIn(padsBegin4D_VPU3720),
            testing::ValuesIn(padsEnd4D_VPU3720),
            testing::ValuesIn(argPadValue_VPU3720),
            testing::Values(ngraph::helpers::PadMode::CONSTANT),
            testing::ValuesIn(netPrecisions),
            testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
            testing::Values(std::vector<size_t>{1, 5, 10, 11}),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    INSTANTIATE_TEST_SUITE_P(
            smoke_Pad_Const_MLIR_VPU3720,
            KmbPadLayerTest_MLIR_VPU3720,
            pad4DConstparams_VPU3720,
            KmbPadLayerTest_MLIR_VPU3720::getTestCaseName
    );

    const auto precommit_pad4DConstparams_VPU3720 = testing::Combine(
            testing::Values(std::vector<int64_t>({4, 2, 1, 3})),
            testing::Values(std::vector<int64_t>({5, 2, 6, 1})),
            testing::Values(-1.f),
            testing::Values(ngraph::helpers::PadMode::CONSTANT),
            testing::ValuesIn(netPrecisions),
            testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Layout::NCHW),
            testing::Values(std::vector<size_t>{1, 5, 10, 11}),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    INSTANTIATE_TEST_SUITE_P(
            smoke_precommit_Pad_Const_MLIR_VPU3720,
            KmbPadLayerTest_MLIR_VPU3720,
            precommit_pad4DConstparams_VPU3720,
            KmbPadLayerTest_MLIR_VPU3720::getTestCaseName
    );

    const auto pad4Dparams_VPU3720 = testing::Combine(
            testing::ValuesIn(padsBegin4D_VPU3720),
            testing::ValuesIn(padsEnd4D_VPU3720),
            testing::Values(0),
            testing::ValuesIn(padMode),
            testing::ValuesIn(netPrecisions),
            testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC),
            testing::Values(std::vector<size_t>{1, 5, 10, 11}),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    INSTANTIATE_TEST_SUITE_P(
            smoke_Pad_MLIR_VPU3720,
            KmbPadLayerTest_MLIR_VPU3720,
            pad4Dparams_VPU3720,
            KmbPadLayerTest_MLIR_VPU3720::getTestCaseName
    );

    const auto precommit_pad4Dparams_VPU3720 = testing::Combine(
            testing::Values(std::vector<int64_t>({0, 0, 2, 0})),
            testing::Values(std::vector<int64_t>({0, 1, 0, 3})),
            testing::Values(0),
            testing::Values(ngraph::helpers::PadMode::EDGE),
            testing::ValuesIn(netPrecisions),
            testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Precision::FP16),
            testing::Values(InferenceEngine::Layout::NCHW),
            testing::Values(std::vector<size_t>{1, 5, 10, 11}),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    INSTANTIATE_TEST_SUITE_P(
            smoke_precommit_Pad_MLIR_VPU3720,
            KmbPadLayerTest_MLIR_VPU3720,
            precommit_pad4Dparams_VPU3720,
            KmbPadLayerTest_MLIR_VPU3720::getTestCaseName
    );

}  // namespace
