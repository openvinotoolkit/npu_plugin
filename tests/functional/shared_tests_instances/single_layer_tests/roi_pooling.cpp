// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/roi_pooling.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class KmbROIPoolingLayerTest : public ROIPoolingLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
        // SkipBeforeInfer() hangs KMB-board.
        // [Track number: S#44493]
        void SkipBeforeImport() override {
            throw LayerTestsUtils::KmbSkipTestException("layer test networks hang the board "
                                                        "if used SkipBeforeInfer()");
        }
    };

    TEST_P(KmbROIPoolingLayerTest, basicTest) {
        Run();
    }

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

const std::vector<std::vector<size_t>> inShapes = {
        {1, 3, 8, 8},
        {3, 4, 50, 50}
};

const std::vector<std::vector<size_t>> pooledShapes_max = {
        {1, 1},
        {2, 2},
        {3, 3},
        {6, 6}
};

const std::vector<std::vector<size_t>> pooledShapes_bilinear = {
        {2, 2},
        {3, 3},
        {6, 6}
};

const std::vector<std::vector<size_t>> coordShapes = {
        {1, 5},
        {3, 5},
        {5, 5}
};

const std::vector<InferenceEngine::Precision> netPRCs = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32
};

const std::vector<float> spatial_scales = {0.625f, 1.f};

const auto test_ROIPooling_max = ::testing::Combine(
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(coordShapes),
        ::testing::ValuesIn(pooledShapes_max),
        ::testing::ValuesIn(spatial_scales),
        ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_MAX),
        ::testing::ValuesIn(netPRCs),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

const auto test_ROIPooling_bilinear = ::testing::Combine(
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(coordShapes),
        ::testing::ValuesIn(pooledShapes_bilinear),
        ::testing::Values(spatial_scales[1]),
        ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_BILINEAR),
        ::testing::ValuesIn(netPRCs),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_TestsROIPooling_max, KmbROIPoolingLayerTest,
                        test_ROIPooling_max, KmbROIPoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke_TestsROIPooling_bilinear, KmbROIPoolingLayerTest,
                        test_ROIPooling_bilinear, KmbROIPoolingLayerTest::getTestCaseName);
