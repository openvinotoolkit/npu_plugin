// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/minimum_maximum.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbMaxMinLayerTest: public MaxMinLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        if (isCompilerMCM()) {
            /// now mcm compiler supports max op (min will be converted to max by IE) - MR #2602
            // throw LayerTestsUtils::KmbSkipTestException("Unsupported operation in MCM compiler [Track number: S#43484]");
        }
    }
};

TEST_P(KmbMaxMinLayerTest, CompareWithRefs) {
    Run();
}

// [Track number: E#14809]
// TEST_P(KmbMaxMinLayerTest, CompareWithRefs_MLIR) {
//     useCompilerMLIR();
//     Run();
// }

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<std::vector<std::vector<size_t>>> inShapes4D = {
        /// TODO: https://jira.devtools.intel.com/browse/EISW-13808
        /// Currently compiler not support a constant input AS default cmajor layout  
//        {{1,64,32,32}, {1,64,32,32}},
//        {{1, 1, 1, 3}, {1}}
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
};

const std::vector<ngraph::helpers::MinMaxOpType> opType = {
        ngraph::helpers::MinMaxOpType::MINIMUM,
        ngraph::helpers::MinMaxOpType::MAXIMUM,
};

const std::vector<ngraph::helpers::InputLayerType> inputType = {
        ngraph::helpers::InputLayerType::CONSTANT
};

const std::vector<InferenceEngine::Layout> layout4D = {
        InferenceEngine::Layout::NCHW,
        InferenceEngine::Layout::NHWC
};

INSTANTIATE_TEST_SUITE_P(smoke_maximum_4D, KmbMaxMinLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes4D),
                                ::testing::ValuesIn(opType),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::ValuesIn(layout4D),
                                ::testing::ValuesIn(layout4D),
                                ::testing::ValuesIn(inputType),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbMaxMinLayerTest::getTestCaseName);

const std::vector<std::vector<std::vector<size_t>>> inShapes3D = {
        /// TODO: https://jira.devtools.intel.com/browse/EISW-13808
//        {{1, 2, 4}, {1}}
};

INSTANTIATE_TEST_SUITE_P(smoke_maximum_3D, KmbMaxMinLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes3D),
                                ::testing::ValuesIn(opType),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputType),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbMaxMinLayerTest::getTestCaseName);

const std::vector<std::vector<std::vector<size_t>>> inShapesScalar = {
        /// test scalar constant input for case MAX(x, scalar_threshold)
        {{32}, {1}}
};

INSTANTIATE_TEST_SUITE_P(smoke_maximum_scalar, KmbMaxMinLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapesScalar),
                                ::testing::ValuesIn(opType),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputType),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbMaxMinLayerTest::getTestCaseName);

}  // namespace
