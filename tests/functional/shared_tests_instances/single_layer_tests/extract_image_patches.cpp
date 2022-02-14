// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/extract_image_patches.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbExtractImagePatchesTest: public ExtractImagePatchesTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

TEST_P(KmbExtractImagePatchesTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

// using namespace ngraph::helpers;
// using ngraph::op::PadType;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
   InferenceEngine::Precision::FP16,
};

// data the 4-D tensor of type T with shape [batch, depth, in_rows, in_cols].

/*
    //  Input  : 1x1x10x10
    //  Output : 1x9x2x2
    auto batch = 1;
    auto depth = 1;
    auto in_rows = 10;
    auto in_cols = 10;
    std::vector<unsigned int> sizes = {3, 3};
    std::vector<unsigned int> strides = {5, 5};
    std::vector<unsigned int> rates = {1, 1};
    std::string auto_pad = "valid"; 
    
*/

const std::vector<std::vector<size_t>> inputShape = {{1, 1, 10, 10}};//{{1, 1, 10, 10}, {1, 3, 10, 10}};
const std::vector<std::vector<size_t>> sizes = {{3, 3}};//{{2, 2}, {3, 3}, {4, 4}, {1, 3}, {4, 2}};
const std::vector<std::vector<size_t>> strides = {{5, 5}};//{{3, 3}, {5, 5}, {9, 9}, {1, 3}, {6, 2}};
const std::vector<std::vector<size_t>> rates = {{1, 1}};//{{1, 1}, {1, 2}, {2, 1}, {2, 2}};
// const std::vector<PadType> paddingType = {PadType::VALID, PadType::SAME_UPPER, PadType::SAME_LOWER};

const auto testExtractImagePatchesParams = ::testing::Combine(
       ::testing::ValuesIn(inputShape),
       ::testing::ValuesIn(sizes),
       ::testing::ValuesIn(strides),
       ::testing::ValuesIn(rates),
       ::testing::Values(ngraph::op::PadType::VALID),
       ::testing::ValuesIn(netPrecisions),
       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
       ::testing::Values(InferenceEngine::Layout::ANY),
       ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_ExtractImagePatches,
                        KmbExtractImagePatchesTest,
                        testExtractImagePatchesParams,
                        KmbExtractImagePatchesTest::getTestCaseName);

}  // namespace
