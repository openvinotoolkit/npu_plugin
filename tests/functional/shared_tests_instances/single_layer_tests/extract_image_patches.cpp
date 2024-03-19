//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/extract_image_patches.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class ExtractImagePatchesTestCommon :
        public ExtractImagePatchesTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class ExtractImagePatchesTest_NPU3700 : public ExtractImagePatchesTestCommon {};
class ExtractImagePatchesTest_NPU3720 : public ExtractImagePatchesTestCommon {};

TEST_P(ExtractImagePatchesTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}
TEST_P(ExtractImagePatchesTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
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

const std::vector<ngraph::op::PadType> paddingType = {ngraph::op::PadType::VALID, ngraph::op::PadType::SAME_UPPER,
                                                      ngraph::op::PadType::SAME_LOWER};
const std::vector<std::vector<size_t>> inputShape = {{1, 3, 10, 10}};
const std::vector<std::vector<size_t>> sizes = {{3, 3}};
const std::vector<std::vector<size_t>> strides = {{5, 5}};
const std::vector<std::vector<size_t>> rates = {{2, 2}};

const auto testExtractImagePatchesParams = ::testing::Combine(
        ::testing::ValuesIn(inputShape), ::testing::ValuesIn(sizes), ::testing::ValuesIn(strides),
        ::testing::ValuesIn(rates), ::testing::ValuesIn(paddingType), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_ExtractImagePatches, ExtractImagePatchesTest_NPU3700, testExtractImagePatchesParams,
                        ExtractImagePatchesTest::getTestCaseName);

// FP16
const auto test1ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1, 10, 10}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{3, 3}}),          // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{5, 5}}),          // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}, {2, 2}}),  // rates
                           ::testing::Values(ngraph::op::PadType::VALID),                          // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                       // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke1_ExtractImagePatches, ExtractImagePatchesTest_NPU3720, test1ExtractImagePatchesParams,
                        ExtractImagePatchesTest::getTestCaseName);

const auto test2ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1, 10, 10}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{4, 4}}),          // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{8, 8}}),          // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),          // rates
                           ::testing::Values(ngraph::op::PadType::VALID),                          // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                       // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke2_ExtractImagePatches, ExtractImagePatchesTest_NPU3720, test2ExtractImagePatchesParams,
                        ExtractImagePatchesTest::getTestCaseName);

const auto test3ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1, 10, 10}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{4, 4}}),          // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{9, 9}}),          // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}, {2, 2}}),  // rates
                           ::testing::Values(ngraph::op::PadType::SAME_UPPER),                     // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                       // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke3_ExtractImagePatches, ExtractImagePatchesTest_NPU3720, test3ExtractImagePatchesParams,
                        ExtractImagePatchesTest::getTestCaseName);

const auto test4ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 2, 5, 5}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),        // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{3, 3}}),        // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),        // rates
                           ::testing::Values(ngraph::op::PadType::VALID),                        // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),                  // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                  // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                  // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                     // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke4_ExtractImagePatches, ExtractImagePatchesTest_NPU3720, test4ExtractImagePatchesParams,
                        ExtractImagePatchesTest::getTestCaseName);

const auto test5ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 64, 26, 26}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),           // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),           // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),           // rates
                           ::testing::Values(ngraph::op::PadType::SAME_LOWER),                      // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),                     // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                     // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                     // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                        // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke5_ExtractImagePatches, ExtractImagePatchesTest_NPU3720, test5ExtractImagePatchesParams,
                        ExtractImagePatchesTest::getTestCaseName);

const auto test6ExtractImagePatchesParams = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 1, 10, 10}}),                // input shape
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{3, 3}}),                        // kernel size
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{5, 5}}),                        // strides
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),                        // rates
        ::testing::Values(ngraph::op::PadType::SAME_UPPER, ngraph::op::PadType::SAME_LOWER),  // pad type
        ::testing::Values(InferenceEngine::Precision::FP16),                                  // Network precision
        ::testing::Values(InferenceEngine::Precision::FP16),                                  // Input precision
        ::testing::Values(InferenceEngine::Precision::FP16),                                  // Output precision
        ::testing::Values(InferenceEngine::Layout::NCHW),                                     // Input layout
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke6_ExtractImagePatches, ExtractImagePatchesTest_NPU3720, test6ExtractImagePatchesParams,
                        ExtractImagePatchesTest::getTestCaseName);

const auto test7ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 3, 13, 37}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 3}}),          // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 4}}),          // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),          // rates
                           ::testing::Values(ngraph::op::PadType::VALID, ngraph::op::PadType::SAME_LOWER),  // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),  // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),  // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),  // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),     // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke7_ExtractImagePatches, ExtractImagePatchesTest_NPU3720, test7ExtractImagePatchesParams,
                        ExtractImagePatchesTest::getTestCaseName);

const auto test8ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 3, 13, 37}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 3}}),          // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 4}}),          // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),          // rates
                           ::testing::Values(ngraph::op::PadType::SAME_UPPER),                     // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                       // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke8_ExtractImagePatches, ExtractImagePatchesTest_NPU3720, test8ExtractImagePatchesParams,
                        ExtractImagePatchesTest::getTestCaseName);

// I32
const auto test9ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 64, 26, 26}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),           // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),           // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),           // rates
                           ::testing::ValuesIn(paddingType),                                        // pad type
                           ::testing::Values(InferenceEngine::Precision::I32),                      // Network precision
                           ::testing::Values(InferenceEngine::Precision::I32),                      // Input precision
                           ::testing::Values(InferenceEngine::Precision::I32),                      // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                        // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke9_ExtractImagePatches, ExtractImagePatchesTest_NPU3720, test9ExtractImagePatchesParams,
                        ExtractImagePatchesTest::getTestCaseName);

const auto testPrecommitExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 3, 10, 10}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{3, 3}}),          // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{5, 5}}),          // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),          // rates
                           ::testing::Values(ngraph::op::PadType::VALID),                          // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                       // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_precommit_ExtractImagePatches, ExtractImagePatchesTest_NPU3720,
                        testPrecommitExtractImagePatchesParams, ExtractImagePatchesTest::getTestCaseName);

}  // namespace
