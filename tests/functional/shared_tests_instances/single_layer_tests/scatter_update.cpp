// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/scatter_update.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class ScatterUpdateLayerTestCommon :
        public ScatterUpdateLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class ScatterUpdateLayerTest_NPU3700 : public ScatterUpdateLayerTestCommon {};
class ScatterUpdateLayerTest_NPU3720 : public ScatterUpdateLayerTestCommon {};

TEST_P(ScatterUpdateLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(ScatterUpdateLayerTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(ScatterUpdateLayerTest_NPU3720, HW) {
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
const auto params = testing::Combine(
        testing::ValuesIn(ScatterUpdateLayerTest::combineShapes(axesShapeInShape)), testing::ValuesIn(scatterIndices),
        testing::Values(InferenceEngine::Precision::FP16), testing::Values(InferenceEngine::Precision::I32),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_ScatterUpdate, ScatterUpdateLayerTest_NPU3700, params,
                         ScatterUpdateLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ScatterUpdate, ScatterUpdateLayerTest_NPU3720, params,
                         ScatterUpdateLayerTest_NPU3720::getTestCaseName);

}  // namespace
