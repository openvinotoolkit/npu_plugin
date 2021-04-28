// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/convert.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbConvertLayerTest: public ConvertLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SetUp() override {
        std::tie(
            std::ignore /*shape*/,
            inPrc, outPrc,
            std::ignore /*inLayout*/, std::ignore /*outLayout*/,
            std::ignore /*deviceName*/
        ) = GetParam();

        ConvertLayerTest::SetUp();
    }

    void SkipBeforeLoad() override {
        if (inPrc == outPrc && isCompilerMCM()) {
            throw LayerTestsUtils::KmbSkipTestException("Same input/output precision not supported for MCM");
        }
        if (outPrc == InferenceEngine::Precision::FP32 && isCompilerMCM()) {
            throw LayerTestsUtils::KmbSkipTestException("FP32 output issue for MCM (bug: E#9603");
        }
        if (inPrc == InferenceEngine::Precision::U8 &&
            (outPrc == InferenceEngine::Precision::FP32 || outPrc == InferenceEngine::Precision::FP16) &&
            isCompilerMCM()) {
            throw LayerTestsUtils::KmbSkipTestException("FP <-> U8 issue for MCM (bug: E#9602");
        }
        if ((inPrc == InferenceEngine::Precision::FP32 || inPrc == InferenceEngine::Precision::FP16) &&
            outPrc == InferenceEngine::Precision::U8 &&
            isCompilerMCM()) {
            throw LayerTestsUtils::KmbSkipTestException("FP <-> U8 issue for MCM (bug: E#9602");
        }
        if (inPrc == InferenceEngine::Precision::I8 ||
            outPrc == InferenceEngine::Precision::I8) {
            throw LayerTestsUtils::KmbSkipTestException("I8 input/output precision is not supported");
        }
        if (outPrc == InferenceEngine::Precision::U8 && isCompilerMLIR()) {
            throw LayerTestsUtils::KmbSkipTestException("U8 output precision is not supported for MLIR");
        }
    }
};

TEST_P(KmbConvertLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbConvertLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<std::vector<size_t>> inShape = {{1, 2, 3, 4}};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::I8,
};

INSTANTIATE_TEST_CASE_P(smoke_NoReshape, KmbConvertLayerTest,
                        ::testing::Combine(
                            ::testing::Values(inShape),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvertLayerTest::getTestCaseName);

}  // namespace
