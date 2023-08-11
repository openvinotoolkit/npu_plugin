// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/scatter_update.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXScatterUpdateLayerTest :
        public ScatterUpdateLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXScatterUpdateLayerTest_VPU3700 : public VPUXScatterUpdateLayerTest {};
class VPUXScatterUpdateLayerTest_VPU3720 : public VPUXScatterUpdateLayerTest {};

TEST_P(VPUXScatterUpdateLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXScatterUpdateLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
// map<inputShape, map<indicesShape, axis>>
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape{
        {{10, 16, 12, 15}, {{{8}, {0, -2}}}}};

const std::vector<std::vector<int64_t>> scatterIndices = {{0, 2, 4, 6, 1, 3, 5, 7}};

INSTANTIATE_TEST_SUITE_P(smoke_ScatterUpdate, VPUXScatterUpdateLayerTest_VPU3700,
                         testing::Combine(testing::ValuesIn(ScatterUpdateLayerTest::combineShapes(axesShapeInShape)),
                                          testing::ValuesIn(scatterIndices),
                                          testing::Values(InferenceEngine::Precision::FP16),
                                          testing::Values(InferenceEngine::Precision::I32),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXScatterUpdateLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ScatterUpdate, VPUXScatterUpdateLayerTest_VPU3720,
                         testing::Combine(testing::ValuesIn(ScatterUpdateLayerTest::combineShapes(axesShapeInShape)),
                                          testing::ValuesIn(scatterIndices),
                                          testing::Values(InferenceEngine::Precision::FP16),
                                          testing::Values(InferenceEngine::Precision::I32),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXScatterUpdateLayerTest_VPU3720::getTestCaseName);

}  // namespace
