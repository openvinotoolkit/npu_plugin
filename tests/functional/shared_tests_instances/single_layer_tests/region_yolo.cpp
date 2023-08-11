// Copyright (C) 2019-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/region_yolo.hpp"
#include <vector>
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {
class VPUXRegionYoloLayerTest : public RegionYoloLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXRegionYoloLayerTest_VPU3700 : public VPUXRegionYoloLayerTest {};

class VPUXRegionYoloLayerTest_VPU3720 : public VPUXRegionYoloLayerTest {};

TEST_P(VPUXRegionYoloLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXRegionYoloLayerTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::Shape> inputShapes = {ngraph::Shape{1, 125, 13, 13}};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_CASE_P(smoke_RegionYolo, VPUXRegionYoloLayerTest_VPU3700,
                        testing::Combine(testing::ValuesIn(inputShapes),
                                         testing::Values(20),                               // classes
                                         testing::Values(4),                                // coords
                                         testing::Values(5),                                // num_regions
                                         testing::Values(false, true),                      // do_softmax
                                         testing::Values(std::vector<int64_t>({0, 1, 2})),  // mask
                                         testing::Values(1),                                // start_axis
                                         testing::Values(3),                                // end_axis
                                         testing::ValuesIn(netPrecisions),
                                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXRegionYoloLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_RegionYolo, VPUXRegionYoloLayerTest_VPU3720,
                        testing::Combine(testing::ValuesIn(inputShapes),
                                         testing::Values(20),                                     // classes
                                         testing::Values(4),                                      // coords
                                         testing::Values(5),                                      // num_regions
                                         testing::Values(false, true),                            // do_softmax
                                         testing::Values(std::vector<int64_t>({0, 1, 2, 3, 4})),  // mask
                                         testing::Values(1),                                      // start_axis
                                         testing::Values(3),                                      // end_axis
                                         testing::ValuesIn(netPrecisions),
                                         testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        VPUXRegionYoloLayerTest::getTestCaseName);

}  // namespace
