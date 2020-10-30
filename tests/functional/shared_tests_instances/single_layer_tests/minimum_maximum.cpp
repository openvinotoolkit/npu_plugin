// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/minimum_maximum.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

// using namespace LayerTestsDefinitions;

namespace LayerTestsDefinitions {

class KmbMaxMinLayerTest: public MaxMinLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeImport() override {
        throw LayerTestsUtils::KmbSkipTestException("layer test networks hang the board");
    }
};

TEST_P(KmbMaxMinLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<std::vector<std::vector<size_t>>> inShapes = {
        {{1,2,2,2}, {1,2,2,2}},
        {{1,64,32,32}, {1,64,32,32}},
        {{1, 1, 1, 3}, {1}},
        {{1, 2, 4}, {1}},
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

INSTANTIATE_TEST_CASE_P(maximum, KmbMaxMinLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes),
                                ::testing::ValuesIn(opType),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputType),
                                ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                        KmbMaxMinLayerTest::getTestCaseName);

}  // namespace
