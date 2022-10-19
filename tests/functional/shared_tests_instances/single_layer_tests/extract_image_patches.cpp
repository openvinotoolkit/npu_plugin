//
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

using ngraph::op::PadType;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
   InferenceEngine::Precision::FP16, 
   InferenceEngine::Precision::I32,
   InferenceEngine::Precision::FP32,
   InferenceEngine::Precision::U8,
};

const std::vector<ngraph::op::PadType> paddingType = {ngraph::op::PadType::VALID, ngraph::op::PadType::SAME_UPPER, ngraph::op::PadType::SAME_LOWER};
const std::vector<std::vector<size_t>> inputShape = {{1, 3, 10, 10}};
const std::vector<std::vector<size_t>> sizes = {{3, 3}};
const std::vector<std::vector<size_t>> strides = {{5, 5}};
const std::vector<std::vector<size_t>> rates =  {{2, 2}};

const auto testExtractImagePatchesParams = ::testing::Combine(
       ::testing::ValuesIn(inputShape),
       ::testing::ValuesIn(sizes),
       ::testing::ValuesIn(strides),
       ::testing::ValuesIn(rates),
       ::testing::ValuesIn(paddingType),
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
