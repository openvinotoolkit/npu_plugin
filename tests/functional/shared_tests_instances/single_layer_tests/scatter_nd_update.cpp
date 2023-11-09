//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/scatter_ND_update.hpp"
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXScatterNDUpdateLayerTest :
        public ScatterNDUpdateLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class VPUXScatterNDUpdateLayerTest_VPU3700 : public VPUXScatterNDUpdateLayerTest {};
class VPUXScatterNDUpdateLayerTest_VPU3720 : public VPUXScatterNDUpdateLayerTest {};

TEST_P(VPUXScatterNDUpdateLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXScatterNDUpdateLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

// map<inputShape map<indicesShape, indicesValue>>
// updateShape is gotten from inputShape and indicesShape
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<size_t>>> sliceSelectInShape{
        {{1}, {{{1, 1}, {0}}}},
        {{8}, {{{4, 1}, {4, 3, 1, 7}}}},
        {{1, 32, 1},
         {{{1, 3, 1, 3}, {0, 10, 0, 0, 11, 0, 0, 12, 0}},
          {{1, 3, 1, 3}, {0, 0, 0, 0, 1, 0, 0, 2, 0}},
          {{1, 3, 1, 3}, {0, 29, 0, 0, 30, 0, 0, 31, 0}}}},
        {{8}, {{{4, 1}, {4, 3, 1, 7}}}},
        {{4, 4, 4}, {{{2, 1}, {0, 2}}, {{2, 1}, {1, 2}}, {{2, 2, 2}, {0, 0, 2, 2, 1, 1, 3, 3}}}},
        {{3, 3, 3},
         {{{2, 1}, {0, 2}},
          {{2, 2, 3}, {0, 0, 0, 2, 2, 2, 1, 0, 0, 1, 2, 2}},
          {{2, 2}, {0, 0, 2, 2}},
          {{2, 3}, {0, 0, 0, 2, 2, 2}}}}};

const auto params = testing::Combine(testing::ValuesIn(ScatterNDUpdateLayerTest::combineShapes(sliceSelectInShape)),
                                     testing::Values(InferenceEngine::Precision::FP16),  // network
                                     testing::Values(InferenceEngine::Precision::I32),   // indices
                                     testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_ScatterNDUpdate, VPUXScatterNDUpdateLayerTest_VPU3700, params,
                         VPUXScatterNDUpdateLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ScatterNDUpdate_VPU3720, VPUXScatterNDUpdateLayerTest_VPU3720, params,
                         VPUXScatterNDUpdateLayerTest::getTestCaseName);

// map<inputShape map<indicesShape, indicesValue>>
// updateShape is gotten from inputShape and indicesShape
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<size_t>>> precommit_sliceSelectInShape{
        // {{2, 3}, {{{1, 2}, {1, 3}}}}, C#108289
        {{2, 3}, {{{1, 2}, {1, 2}}}},
};

const auto precommit_params =
        testing::Combine(testing::ValuesIn(ScatterNDUpdateLayerTest::combineShapes(precommit_sliceSelectInShape)),
                         testing::Values(InferenceEngine::Precision::FP16),  // network
                         testing::Values(InferenceEngine::Precision::I32),   // indices
                         testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ScatterNDUpdate_VPU3720, VPUXScatterNDUpdateLayerTest_VPU3720,
                         precommit_params, VPUXScatterNDUpdateLayerTest::getTestCaseName);

}  // namespace
