// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/conversion.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbConversionLayerTest: public ConversionLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SetUp() override {
        std::tie(
            std::ignore /*conversionOpType*/,
            std::ignore /*shape*/,
            inPrc, outPrc,
            std::ignore /*inLayout*/, std::ignore /*outLayout*/,
            std::ignore /*deviceName*/
        ) = GetParam();

        ConversionLayerTest::SetUp();
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
        if (inPrc == InferenceEngine::Precision::FP32 && outPrc == InferenceEngine::Precision::FP16 &&
            isCompilerMCM()) {
            throw LayerTestsUtils::KmbSkipTestException("FP <-> FP16 issue for MCM (bug: E#28335");
        }
    }
};

TEST_P(KmbConversionLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbConversionLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::helpers::ConversionTypes> conversionOpTypes = {
    ngraph::helpers::ConversionTypes::CONVERT,
    ngraph::helpers::ConversionTypes::CONVERT_LIKE,
};

const std::vector<std::vector<size_t>> inShape = {{1, 2, 3, 4}};

// Precision I8 was deleted from netPrecisions because it is not supported by run-time and compiler.
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::U8
};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, KmbConversionLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(conversionOpTypes),
                            ::testing::Values(inShape),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Layout::NHWC),
                            ::testing::Values(InferenceEngine::Layout::NHWC),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConversionLayerTest::getTestCaseName);

}  // namespace
