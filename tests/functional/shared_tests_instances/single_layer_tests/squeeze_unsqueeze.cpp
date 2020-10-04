// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/squeeze_unsqueeze.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbSqueezeUnsqueezeLayerTest: public SqueezeUnsqueezeLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

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
INSTANTIATE_TEST_CASE_P(DISABLED_Basic, KmbSqueezeUnsqueezeLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(CommonTestUtils::combineParams(axesVectors)),
                            ::testing::ValuesIn(opTypes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                        SqueezeUnsqueezeLayerTest::getTestCaseName);
}  // namespace
