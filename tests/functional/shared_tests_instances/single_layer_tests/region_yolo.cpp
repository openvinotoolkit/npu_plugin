// Copyright (C) 2019-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/region_yolo.hpp"
#include <vector>
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class RegionYoloLayerTestCommon : public RegionYoloLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};
class RegionYoloLayerTest_NPU3700 : public RegionYoloLayerTestCommon {};
class RegionYoloLayerTest_NPU3720 : public RegionYoloLayerTestCommon {};

TEST_P(RegionYoloLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(RegionYoloLayerTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ngraph::Shape> inputShapes = {ngraph::Shape{1, 125, 13, 13}};

const std::vector<ngraph::Shape> inputShapesPrecommit = {ngraph::Shape{1, 27, 26, 26}};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_CASE_P(smoke_RegionYolo, RegionYoloLayerTest_NPU3700,
                        testing::Combine(testing::ValuesIn(inputShapes),
                                         testing::Values(20),                               // classes
                                         testing::Values(4),                                // coords
                                         testing::Values(5),                                // num_regions
                                         testing::Values(false, true),                      // do_softmax
                                         testing::Values(std::vector<int64_t>({0, 1, 2})),  // mask
                                         testing::Values(1),                                // start_axis
                                         testing::Values(3),                                // end_axis
                                         testing::ValuesIn(netPrecisions),
                                         testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                        RegionYoloLayerTestCommon::getTestCaseName);

const auto regionYoloParams = ::testing::Combine(testing::ValuesIn(inputShapes),
                                                 testing::Values(20),                                     // classes
                                                 testing::Values(4),                                      // coords
                                                 testing::Values(5),                                      // num_regions
                                                 testing::Values(false, true),                            // do_softmax
                                                 testing::Values(std::vector<int64_t>({0, 1, 2, 3, 4})),  // mask
                                                 testing::Values(1),                                      // start_axis
                                                 testing::Values(3),                                      // end_axis
                                                 testing::ValuesIn(netPrecisions),
                                                 testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto regionYoloPrecommitParams = ::testing::Combine(testing::ValuesIn(inputShapesPrecommit),
                                                          testing::Values(4),      // classes
                                                          testing::Values(4),      // coords
                                                          testing::Values(9),      // num_regions
                                                          testing::Values(false),  // do_softmax
                                                          testing::Values(std::vector<int64_t>({0, 1, 2})),  // mask
                                                          testing::Values(1),  // start_axis
                                                          testing::Values(3),  // end_axis
                                                          testing::ValuesIn(netPrecisions),
                                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_RegionYolo, RegionYoloLayerTest_NPU3720, regionYoloParams,
                        RegionYoloLayerTestCommon::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_RegionYolo, RegionYoloLayerTest_NPU3720, regionYoloPrecommitParams,
                        RegionYoloLayerTestCommon::getTestCaseName);

}  // namespace
