// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include <common/functions.h>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/deformable_psroi_pooling.hpp"

namespace LayerTestsDefinitions {

class KmbDeformablePSROIPoolingLayerTest :
        public DeformablePSROIPoolingLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbDeformablePSROIPoolingLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const auto deformablePSROIParams = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 8, 16, 16}, {1, 8, 67, 32}}),  // data input shape
        ::testing::Values(std::vector<size_t>{10, 5}),                                          // rois input shape
        // Empty offsets shape means test without optional third input
        ::testing::ValuesIn(std::vector<std::vector<size_t>>{{}, {10, 2, 2, 2}}),                // offsets input shape
        ::testing::Values(2),                                                                    // output_dim
        ::testing::Values(2),                                                                    // group_size
        ::testing::ValuesIn(std::vector<float>{1.0, 0.0625}),                                    // spatial scale
        ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{1, 1}, {2, 2}, {3, 3}, {4, 4}}),  // spatial_bins_x_y
        ::testing::ValuesIn(std::vector<float>{0.0, 0.01, 0.1}),                                 // trans_std
        ::testing::Values(2));

const auto deformablePSROICases_test_params =
        ::testing::Combine(deformablePSROIParams,
                           ::testing::Values(InferenceEngine::Precision::FP32),            // Net precision
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));  // Device name

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_TestsDeformablePSROIPooling, KmbDeformablePSROIPoolingLayerTest,
                         deformablePSROICases_test_params, KmbDeformablePSROIPoolingLayerTest::getTestCaseName);

const auto deformablePSROIParams_advanced =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 441, 8, 8}}),  // data input shape
                           ::testing::Values(std::vector<size_t>{30, 5}),                          // rois input shape
                           ::testing::Values(std::vector<size_t>{30, 2, 3, 3}),             // offsets input shape
                           ::testing::Values(49),                                           // output_dim
                           ::testing::Values(3),                                            // group_size
                           ::testing::ValuesIn(std::vector<float>{0.0625}),                 // spatial scale
                           ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{4, 4}}),  // spatial_bins_x_y
                           ::testing::ValuesIn(std::vector<float>{0.1}),                    // trans_std
                           ::testing::Values(3));                                           // part_size

const auto deformablePSROICases_test_params_advanced =
        ::testing::Combine(deformablePSROIParams_advanced,
                           ::testing::Values(InferenceEngine::Precision::FP32),            // Net precision
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));  // Device name

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_TestsDeformablePSROIPooling_advanced, KmbDeformablePSROIPoolingLayerTest,
                         deformablePSROICases_test_params_advanced,
                         KmbDeformablePSROIPoolingLayerTest::getTestCaseName);

const auto deformablePSROIParams_advanced1 =
        ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 441, 8, 8}}),  // data input shape
                           ::testing::Values(std::vector<size_t>{30, 5}),                          // rois input shape
                           ::testing::Values(std::vector<size_t>{}),         // offsets input shape
                           ::testing::Values(49),                            // output_dim
                           ::testing::Values(3),                             // group_size
                           ::testing::ValuesIn(std::vector<float>{0.0625}),  // spatial scale
                           ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{1, 1}, {4, 4}}),  // spatial_bins_x_y
                           ::testing::ValuesIn(std::vector<float>{0.0, 0.1}),                       // trans_std
                           ::testing::Values(3));                                                   // part_size

const auto deformablePSROICases_test_params_advanced1 =
        ::testing::Combine(deformablePSROIParams_advanced1,
                           ::testing::Values(InferenceEngine::Precision::FP32),            // Net precision
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));  // Device name

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_TestsDeformablePSROIPooling_advanced1, KmbDeformablePSROIPoolingLayerTest,
                         deformablePSROICases_test_params_advanced1,
                         KmbDeformablePSROIPoolingLayerTest::getTestCaseName);

}  // namespace
