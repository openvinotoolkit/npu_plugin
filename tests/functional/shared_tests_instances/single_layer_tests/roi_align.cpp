//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include <common/functions.h>
#include "kmb_layer_test.hpp"
#include "single_layer_tests/roi_align.hpp"

namespace LayerTestsDefinitions {

class VPUXROIAlignLayerTest_VPU3700 : public ROIAlignLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
    }
    void SkipBeforeInfer() override {
        if (getBackendName(*getCore()) == "LEVEL0") {
            throw LayerTestsUtils::KmbSkipTestException("Bad results on Level0");
        }
    }
};

TEST_P(VPUXROIAlignLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecision = {InferenceEngine::Precision::FP16};

const std::vector<std::vector<size_t>> inputShape = {{2, 18, 20, 20}, {2, 4, 20, 20}, {2, 4, 20, 40}, {10, 1, 20, 20}};

const std::vector<std::vector<size_t>> coordsShape = {{2, 4}};

const std::vector<int> pooledH = {2};

const std::vector<int> pooledW = {2};

const std::vector<float> spatialScale = {0.625f, 1.0f};

const std::vector<int> poolingRatio = {2};

const std::vector<std::string> poolingMode = {"avg", "max"};

const auto testROIAlignParams =
        testing::Combine(testing::ValuesIn(inputShape), testing::ValuesIn(coordsShape), testing::ValuesIn(pooledH),
                         testing::ValuesIn(pooledW), testing::ValuesIn(spatialScale), testing::ValuesIn(poolingRatio),
                         testing::ValuesIn(poolingMode), testing::ValuesIn(netPrecision),
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_ROIAlign, VPUXROIAlignLayerTest_VPU3700, testROIAlignParams,
                         VPUXROIAlignLayerTest_VPU3700::getTestCaseName);
}  // namespace
