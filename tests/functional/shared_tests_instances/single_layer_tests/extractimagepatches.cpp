// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/extractimagepatches.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbExtractImagePatchesLayerTest: public ExtractImagePatchesLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

TEST_P(KmbExtractImagePatchesLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using ngraph::op::PadType;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
};

//std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = { // ???
//    {{1, 50, 1, 1}, {{}}},
//    {{1, 128, 1, 1}, {{}}},
//};

// data the 4-D tensor of type T with shape [batch, depth, in_rows, in_cols].
// const std::vector<InferenceEngine::SizeVector> inputShapes = {
//     InferenceEngine::SizeVector {1, 1, 10, 10}, 
//     InferenceEngine::SizeVector {1, 3, 10, 10}
// };

//const std::vector<size_t> inputShape = {
//        {1, 1, 10, 10}, 
//        {1, 3, 10, 10}
//};

// const std::vector<std::vector<size_t>> sizes = {{2, 2}, {3, 3}, {4, 4}, {1, 3}, {4, 2}};
// const std::vector<std::vector<size_t>> strides = {{3, 3}, {5, 5}, {9, 9}, {1, 3}, {6, 2}};
// const std::vector<std::vector<size_t>> rates = {{1, 1}, {1, 2}, {2, 1}, {2, 2}};
// const std::vector<PadType> paddingType = {PadType::VALID, PadType::SAME_UPPER, PadType::SAME_LOWER};

const auto testExtractImagePatchesParams = testing::Combine(
        testing::ValuesIn(inputShape),
        testing::ValuesIn(sizes),
        testing::ValuesIn(strides),
        testing::ValuesIn(rates)
        testing::ValuesIn(paddingType),
        testing::ValuesIn(netPrecision),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

//const auto basicCases = testing::Combine(
//    testing::ValuesIn(netPrecisions),
//    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
//    testing::Values(InferenceEngine::Layout::ANY),
//    testing::Values(InferenceEngine::Layout::ANY),
//    testing::ValuesIn(inputShapes),
//    testing::ValuesIn(basic), // ??
//    testing::Values(LayerTestsUtils::testPlatformTargetDevice)
//);

INSTANTIATE_TEST_CASE_P(smoke_ExtractImagePatches, KmbExtractImagePatchesLayerTest, testExtractImagePatchesParams, ExtractImagePatchesLayerTest::getTestCaseName);

}  // namespace