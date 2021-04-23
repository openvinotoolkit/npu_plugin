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
            throw LayerTestsUtils::KmbSkipTestException("Unsupported operation in MCM compiler [Track number: S#43484]");
        }
    }
};

TEST_P(KmbMaxMinLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbMaxMinLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<std::vector<std::vector<size_t>>> inShapes4D = {
        {{1,64,32,32}, {1,64,32,32}},
        {{1, 1, 1, 3}, {1}}
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

INSTANTIATE_TEST_CASE_P(smoke_maximum_4D, KmbMaxMinLayerTest,
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
        {{1, 2, 4}, {1}},
};

INSTANTIATE_TEST_CASE_P(smoke_maximum_3D, KmbMaxMinLayerTest,
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

}  // namespace
