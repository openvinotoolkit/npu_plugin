// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/pad.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class KmbPadLayerTest: public PadLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

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
            //InferenceEngine::Precision::FP32,
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

}  // namespace
