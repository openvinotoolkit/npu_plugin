//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/extract_image_patches.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbExtractImagePatchesTest :
        public ExtractImagePatchesTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbExtractImagePatchesTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}
class VPUXExtractImagePatchesLayerTest_VPU3720 :
        public ExtractImagePatchesTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(VPUXExtractImagePatchesLayerTest_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
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
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_ExtractImagePatches, KmbExtractImagePatchesTest,
                        testExtractImagePatchesParams, KmbExtractImagePatchesTest::getTestCaseName);

//
// VPUX3720 tests
// [Track number: E#61053]
//

// FP16
const auto testVPU37201ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1, 10, 10}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{3, 3}}),          // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{5, 5}}),          // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}, {2, 2}}),  // rates
                           ::testing::Values(ngraph::op::PadType::VALID),                          // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                       // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke1_ExtractImagePatches_VPU3720, VPUXExtractImagePatchesLayerTest_VPU3720,
                        testVPU37201ExtractImagePatchesParams, KmbExtractImagePatchesTest::getTestCaseName);

const auto testVPU37202ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1, 10, 10}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{4, 4}}),          // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{8, 8}}),          // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),          // rates
                           ::testing::Values(ngraph::op::PadType::VALID),                          // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                       // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(DISABLED_smoke2_ExtractImagePatches_VPU3720, VPUXExtractImagePatchesLayerTest_VPU3720,
                        testVPU37202ExtractImagePatchesParams, KmbExtractImagePatchesTest::getTestCaseName);

const auto testVPU37203ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1, 10, 10}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{4, 4}}),          // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{9, 9}}),          // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}, {2, 2}}),  // rates
                           ::testing::Values(ngraph::op::PadType::SAME_UPPER),                     // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                       // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke3_ExtractImagePatches_VPU3720, VPUXExtractImagePatchesLayerTest_VPU3720,
                        testVPU37203ExtractImagePatchesParams, KmbExtractImagePatchesTest::getTestCaseName);

const auto testVPU37204ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 2, 5, 5}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),        // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{3, 3}}),        // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),        // rates
                           ::testing::Values(ngraph::op::PadType::VALID),                        // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),                  // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                  // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                  // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                     // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke4_ExtractImagePatches_VPU3720, VPUXExtractImagePatchesLayerTest_VPU3720,
                        testVPU37204ExtractImagePatchesParams, KmbExtractImagePatchesTest::getTestCaseName);

const auto testVPU37205ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 64, 26, 26}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),           // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),           // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),           // rates
                           ::testing::Values(ngraph::op::PadType::SAME_LOWER),                      // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),                     // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                     // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                     // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                        // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(DISABLED_smoke5_ExtractImagePatches_VPU3720, VPUXExtractImagePatchesLayerTest_VPU3720,
                        testVPU37205ExtractImagePatchesParams, KmbExtractImagePatchesTest::getTestCaseName);

const auto testVPU37206ExtractImagePatchesParams = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 1, 10, 10}}),                // input shape
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{3, 3}}),                        // kernel size
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{5, 5}}),                        // strides
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),                        // rates
        ::testing::Values(ngraph::op::PadType::SAME_UPPER, ngraph::op::PadType::SAME_LOWER),  // pad type
        ::testing::Values(InferenceEngine::Precision::FP16),                                  // Network precision
        ::testing::Values(InferenceEngine::Precision::FP16),                                  // Input precision
        ::testing::Values(InferenceEngine::Precision::FP16),                                  // Output precision
        ::testing::Values(InferenceEngine::Layout::NCHW),                                     // Input layout
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke6_ExtractImagePatches_VPU3720, VPUXExtractImagePatchesLayerTest_VPU3720,
                        testVPU37206ExtractImagePatchesParams, KmbExtractImagePatchesTest::getTestCaseName);

const auto testVPU37207ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 3, 13, 37}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 3}}),          // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 4}}),          // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),          // rates
                           ::testing::Values(ngraph::op::PadType::VALID, ngraph::op::PadType::SAME_LOWER),  // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),  // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),  // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),  // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),     // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(DISABLED_smoke7_ExtractImagePatches_VPU3720, VPUXExtractImagePatchesLayerTest_VPU3720,
                        testVPU37207ExtractImagePatchesParams, KmbExtractImagePatchesTest::getTestCaseName);

const auto testVPU37208ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 3, 13, 37}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 3}}),          // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 4}}),          // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),          // rates
                           ::testing::Values(ngraph::op::PadType::SAME_UPPER),                     // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                       // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke8_ExtractImagePatches_VPU3720, VPUXExtractImagePatchesLayerTest_VPU3720,
                        testVPU37208ExtractImagePatchesParams, KmbExtractImagePatchesTest::getTestCaseName);

// I32
const auto testVPU37209ExtractImagePatchesParams =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 64, 26, 26}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),           // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),           // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 1}}),           // rates
                           ::testing::ValuesIn(paddingType),                                        // pad type
                           ::testing::Values(InferenceEngine::Precision::I32),                      // Network precision
                           ::testing::Values(InferenceEngine::Precision::I32),                      // Input precision
                           ::testing::Values(InferenceEngine::Precision::I32),                      // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                        // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke9_ExtractImagePatches_VPU3720, VPUXExtractImagePatchesLayerTest_VPU3720,
                        testVPU37209ExtractImagePatchesParams, KmbExtractImagePatchesTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
        smoke_precommit_ExtractImagePatches_VPU3720, VPUXExtractImagePatchesLayerTest_VPU3720,
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 3, 10, 10}}),  // input shape
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{3, 3}}),          // kernel size
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{5, 5}}),          // strides
                           ::testing::ValuesIn(std::vector<std::vector<size_t>>{{2, 2}}),          // rates
                           ::testing::Values(ngraph::op::PadType::VALID),                          // pad type
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Network precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Input precision
                           ::testing::Values(InferenceEngine::Precision::FP16),                    // Output precision
                           ::testing::Values(InferenceEngine::Layout::NCHW),                       // Input layout
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
        KmbExtractImagePatchesTest::getTestCaseName);

}  // namespace
