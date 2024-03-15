//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include <common/functions.h>
#include "single_layer_tests/roi_pooling.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class ROIPoolingLayerTestCommon : public ROIPoolingLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void GenerateInputs() override {
        ngraph::helpers::ROIPoolingTypes poolMethod;
        float spatialScale = 0.f;
        std::tie(std::ignore, std::ignore, std::ignore, spatialScale, poolMethod, std::ignore, std::ignore) =
                GetParam();

        inputs.clear();

        auto feat_map_shape = cnnNetwork.getInputShapes().begin()->second;

        const auto is_roi_max_mode = (poolMethod == ngraph::helpers::ROIPoolingTypes::ROI_MAX);

        const int height = is_roi_max_mode ? feat_map_shape[2] / spatialScale : 1;
        const int width = is_roi_max_mode ? feat_map_shape[3] / spatialScale : 1;

        size_t it = 0;
        for (const auto& input : cnnNetwork.getInputsInfo()) {
            const auto& info = input.second;
            InferenceEngine::Blob::Ptr blob;

            if (it == 1) {
                blob = make_blob_with_precision(info->getTensorDesc());
                blob->allocate();
                ov::test::utils::fill_data_roi<InferenceEngine::Precision::FP32>(blob, feat_map_shape[0] - 1, height,
                                                                                 width, 1.0f, is_roi_max_mode);
            } else {
                blob = GenerateInput(*info);
            }
            inputs.push_back(blob);
            it++;
        }
    }
};
class ROIPoolingLayerTest_NPU3700 : public ROIPoolingLayerTestCommon {
    void SkipBeforeInfer() override {
        // Tracking number [E#85137]
        if (getBackendName(*getCore()) == "LEVEL0") {
            throw LayerTestsUtils::VpuSkipTestException("Bad results on Level0");
        }
    }
};

class ROIPoolingLayerTest_NPU3720 : public ROIPoolingLayerTestCommon {};

TEST_P(ROIPoolingLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(ROIPoolingLayerTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

const std::vector<std::vector<size_t>> inShapes = {{1, 3, 8, 8}, {3, 4, 50, 50}};

const std::vector<std::vector<size_t>> pooledShapes_max = {{1, 1}, {2, 2}, {3, 3}, {6, 6}};

const std::vector<std::vector<size_t>> pooledShapes_bilinear = {/*{1, 1},*/ {2, 2}, {3, 3}, {6, 6}};

const std::vector<std::vector<size_t>> coordShapes = {{1, 5}, /*{3, 5}, {5, 5}*/};

const std::vector<InferenceEngine::Precision> netPRCs = {InferenceEngine::Precision::FP16};

const std::vector<float> spatial_scales = {0.625f, 1.f};

const auto test_ROIPooling_max = ::testing::Combine(
        ::testing::ValuesIn(inShapes), ::testing::ValuesIn(coordShapes), ::testing::ValuesIn(pooledShapes_max),
        ::testing::ValuesIn(spatial_scales), ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_MAX),
        ::testing::ValuesIn(netPRCs), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto test_ROIPooling_bilinear = ::testing::Combine(
        ::testing::ValuesIn(inShapes), ::testing::ValuesIn(coordShapes), ::testing::ValuesIn(pooledShapes_bilinear),
        ::testing::Values(spatial_scales[1]), ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_BILINEAR),
        ::testing::ValuesIn(netPRCs), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// --------- NPU3700 ---------
INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_max, ROIPoolingLayerTest_NPU3700, test_ROIPooling_max,
                         ROIPoolingLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_bilinear, ROIPoolingLayerTest_NPU3700, test_ROIPooling_bilinear,
                         ROIPoolingLayerTest_NPU3700::getTestCaseName);

// --------- NPU3720 ---------
INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_max, ROIPoolingLayerTest_NPU3720, test_ROIPooling_max,
                         ROIPoolingLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_bilinear, ROIPoolingLayerTest_NPU3720, test_ROIPooling_bilinear,
                         ROIPoolingLayerTest_NPU3720::getTestCaseName);
