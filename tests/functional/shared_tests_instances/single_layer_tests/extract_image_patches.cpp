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
//using ngraph::op::PadType;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
   InferenceEngine::Precision::FP16,
};

/* valid
{64, 3, 10, 10}, sizes - {3,3}, strides - {5,5}, rates {1,1} => {64, 27, 2, 2}
{64, 3, 10, 10}, sizes - {3,3}, strides - {5,5}, rates {2,2} => {64, 27, 2, 2}
{64, 3, 9, 9}, sizes - {3,3}, strides - {5,5}, rates {2,2} => {64, 27, 1, 1}
*/
const std::vector<std::vector<size_t>> inputShape_v = {  {64, 3, 10, 10} , {64, 3, 9, 9} };
const std::vector<std::vector<size_t>> sizes_v = {  {3, 3} };
const std::vector<std::vector<size_t>> strides_v = {  {5, 5} };
const std::vector<std::vector<size_t>> rates_v =  {  {1, 1} , {2, 2} };
//const std::vector<ngraph::op::PadType> paddingType = {ngraph::op::PadType::VALID, ngraph::op::PadType::SAME_UPPER, ngraph::op::PadType::SAME_LOWER};

/* same_upper
{64, 3, 11, 11}, sizes - {3,3}, strides - {5,5}, rates {1,1} => {64, 27, 3, 3}
{64, 3, 6, 11}, sizes - {3,3}, strides - {5,5}, rates {2,2} => {64, 27, 2, 3}
*/
const std::vector<std::vector<size_t>> inputShape_su = {  {64, 3, 11, 11} , {64, 3, 6, 11}  };
const std::vector<std::vector<size_t>> sizes_su = {  {3, 3} };
const std::vector<std::vector<size_t>> strides_su =  {  {5, 5} };
const std::vector<std::vector<size_t>> rates_su =  {  {1, 1} };

/* same_lower 
{64, 3, 10, 10}, sizes - {3,3}, strides - {5,5}, rates {1,1} => {64, 27, 2, 2}
{64, 3, 9, 9}, sizes - {3,3}, strides - {5,5}, rates {2,2} => {64, 27, 2, 2}
*/
const std::vector<std::vector<size_t>> inputShape_sl = {  {64, 3, 10, 10} , {64, 3, 9, 9}  };
const std::vector<std::vector<size_t>> sizes_sl = {  {3, 3} };
const std::vector<std::vector<size_t>> strides_sl = {  {5, 5} };
const std::vector<std::vector<size_t>> rates_sl =   {  {1, 1} };

const auto testExtractImagePatchesParams_valid = ::testing::Combine(
       ::testing::ValuesIn(inputShape_v),
       ::testing::ValuesIn(sizes_v),
       ::testing::ValuesIn(strides_v),
       ::testing::ValuesIn(rates_v),
       ::testing::Values(ngraph::op::PadType::VALID),
       ::testing::ValuesIn(netPrecisions),
       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
       ::testing::Values(InferenceEngine::Layout::ANY),
       ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

const auto testExtractImagePatchesParams_same_upper = ::testing::Combine(
       ::testing::ValuesIn(inputShape_su),
       ::testing::ValuesIn(sizes_su),
       ::testing::ValuesIn(strides_su),
       ::testing::ValuesIn(rates_su),
       ::testing::Values(ngraph::op::PadType::SAME_UPPER),
       ::testing::ValuesIn(netPrecisions),
       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
       ::testing::Values(InferenceEngine::Layout::ANY),
       ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

const auto testExtractImagePatchesParams_same_lower = ::testing::Combine(
       ::testing::ValuesIn(inputShape_sl),
       ::testing::ValuesIn(sizes_sl),
       ::testing::ValuesIn(strides_sl),
       ::testing::ValuesIn(rates_sl),
       ::testing::Values(ngraph::op::PadType::SAME_LOWER),
       ::testing::ValuesIn(netPrecisions),
       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
       ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
       ::testing::Values(InferenceEngine::Layout::ANY),
       ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_ExtractImagePatches1,
                        KmbExtractImagePatchesTest,
                        testExtractImagePatchesParams_valid,
                        KmbExtractImagePatchesTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ExtractImagePatches2,
                        KmbExtractImagePatchesTest,
                        testExtractImagePatchesParams_same_upper,
                        KmbExtractImagePatchesTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ExtractImagePatches3,
                        KmbExtractImagePatchesTest,
                        testExtractImagePatchesParams_same_lower,
                        KmbExtractImagePatchesTest::getTestCaseName);

}  // namespace
