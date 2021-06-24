// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/topk.hpp"

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbTopKLayerTest: virtual public TopKLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbTopKLayerTest, CompareWithRefs) {
    KmbLayerTestsCommon::Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16
};

const std::vector<int64_t> axes = {0, 1, 2};

const std::vector<int64_t> k = {1, 5, 10};

const std::vector<ngraph::opset4::TopK::Mode> modes = {
        ngraph::opset4::TopK::Mode::MIN,
        ngraph::opset4::TopK::Mode::MAX
};

const std::vector<ngraph::opset4::TopK::SortType> sortTypes = {
        ngraph::opset4::TopK::SortType::NONE,
        ngraph::opset4::TopK::SortType::SORT_INDICES,
        ngraph::opset4::TopK::SortType::SORT_VALUES,
};

// [Track number: S#41824]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_TopK, KmbTopKLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(k),
                ::testing::ValuesIn(axes),
                ::testing::ValuesIn(modes),
                ::testing::ValuesIn(sortTypes),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::FP16),
                ::testing::Values(InferenceEngine::Precision::FP16),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({10, 10, 10})),
                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        TopKLayerTest::getTestCaseName);
}  // namespace
