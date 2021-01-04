// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/squeeze_unsqueeze.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbSqueezeUnsqueezeLayerTest: public SqueezeUnsqueezeLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        if (envConfig.IE_KMB_TESTS_RUN_INFER) {
            throw LayerTestsUtils::KmbSkipTestException("layer test networks hang the board");
        }
    }
};

TEST_P(KmbSqueezeUnsqueezeLayerTest, BasicTest) {
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
std::map<std::vector<size_t>, std::vector<std::vector<int>>> axesVectors = {
    {{1, 1, 1, 1}, {{-1}, {0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {2, 3}, {0, 1, 2}, {0, 2, 3}, {1, 2, 3}, {0, 1, 2, 3}}},
    {{1, 2, 3, 4}, {{0}}},
    {{2, 1, 3, 4}, {{1}}},
    {{1}, {{-1}, {0}}},
    {{1, 2}, {{0}}},
    {{2, 1}, {{1}, {-1}}},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<ngraph::helpers::SqueezeOpType> opTypes = {
    ngraph::helpers::SqueezeOpType::SQUEEZE,
    ngraph::helpers::SqueezeOpType::UNSQUEEZE
};

// Tests fail with similar errors:
// C++ exception with description "Size of dims(1) and format(NHWC) are inconsistent.
// C++ exception with description "Size of dims(3) and format(NHWC) are inconsistent.
// C++ exception with description "Size of dims(5) and format(NHWC) are inconsistent.
// and so on.
// [Track number: S#39977]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_Basic, KmbSqueezeUnsqueezeLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(CommonTestUtils::combineParams(axesVectors)),
                            ::testing::ValuesIn(opTypes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        SqueezeUnsqueezeLayerTest::getTestCaseName);


// Subset of parameters and test for Unsqueeze.
// This subset is used to enable test on Usqueeze.
// Do not forget to remove this subset and test when initial test DISABLED_smoke_Basic will be enabled.

std::map<std::vector<size_t>, std::vector<std::vector<int>>> axesVectors_unsqueeze_pass_mcm = {
    {{1}, {{-1}, {0}}},
    {{1, 2}, {{0}}},
    {{2, 1}, {{1}, {-1}}},
};

const std::vector<ngraph::helpers::SqueezeOpType> opTypes_unsqueeze_pass_mcm = {
    ngraph::helpers::SqueezeOpType::UNSQUEEZE
};

INSTANTIATE_TEST_CASE_P(smoke_Basic_unsqueeze_pass_mcm, KmbSqueezeUnsqueezeLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(CommonTestUtils::combineParams(axesVectors_unsqueeze_pass_mcm)),
                            ::testing::ValuesIn(opTypes_unsqueeze_pass_mcm),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        SqueezeUnsqueezeLayerTest::getTestCaseName);

// End of parameters and test for Unsqueeze.

// Subset of parameters and test for Squeeze.
// This subset is used to enable test on Squeeze.
// Do not forget to remove this subset and test when initial test DISABLED_smoke_Basic will be enabled.

std::map<std::vector<size_t>, std::vector<std::vector<int>>> axesVectors_squeeze_pass_mcm = {
    {{1, 1, 1, 1}, {{-1}, {0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {2, 3}, {0, 1, 2}, {0, 2, 3}, {1, 2, 3}}},
    {{1, 2, 3, 4}, {{0}}},
    {{2, 1, 3, 4}, {{1}}},
    {{1, 2}, {{0}}},
    {{2, 1}, {{1}, {-1}}},
};

const std::vector<ngraph::helpers::SqueezeOpType> opTypes_squeeze_pass_mcm = {
    ngraph::helpers::SqueezeOpType::SQUEEZE
};

INSTANTIATE_TEST_CASE_P(smoke_Basic_squeeze_pass_mcm, KmbSqueezeUnsqueezeLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(CommonTestUtils::combineParams(axesVectors_squeeze_pass_mcm)),
                            ::testing::ValuesIn(opTypes_squeeze_pass_mcm),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        SqueezeUnsqueezeLayerTest::getTestCaseName);

// End of parameters and test for Squeeze.

}  // namespace
