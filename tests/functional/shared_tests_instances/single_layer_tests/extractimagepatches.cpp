// Copyright (C) Intel Corporation
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
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
};

//std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
//    {{1, 50, 1, 1}, {{}}},
//    {{1, 128, 1, 1}, {{}}},
//};

// data the 4-D tensor of type T with shape [batch, depth, in_rows, in_cols].
const std::vector<InferenceEngine::SizeVector> inputShapes = {
    InferenceEngine::SizeVector { 64, 3, 10, 10 },
};

//const std::vector<size_t> inputShape = {
//        { 2, 18, 20, 20 },
//        { 2, 4, 20, 20 },
//        { 2, 4, 20, 40 },
//        { 10, 1, 20, 20 }
//};

//const std::vector<size_t> sizes = {{3, 3}}; // int64_t

//const std::vector<size_t> strides = {{5, 5}}; // int64_t

//const std::vector<size_t> rates = {{1, 1}}; // int64_t

const std::vector<std::string> paddingType = {
            "same_upper",
            "same_lower",
            "valid"
    };


const auto testExtractImagePatchesParams = testing::Combine(
        testing::ValuesIn(inputShape),
        testing::ValuesIn(sizes),
        testing::ValuesIn(strides),
        testing::ValuesIn(rates)
        testing::ValuesIn(paddingType),
        testing::ValuesIn(netPrecision),
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

INSTANTIATE_TEST_CASE_P(smoke_ExtractImagePatches, KmbExtractImagePatchesLayerTest, testExtractImagePatchesParams, ExtractImagePatchesLayerTest::getTestCaseName
);
