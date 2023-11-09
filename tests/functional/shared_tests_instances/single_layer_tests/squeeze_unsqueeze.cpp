//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/squeeze_unsqueeze.hpp"
#include <vector>
#include "common/functions.h"
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXSqueezeUnsqueezeLayerTest :
        public SqueezeUnsqueezeLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
protected:
    void SkipBeforeLoad() override {
        const auto inRank = function->get_parameters().at(0)->get_output_shape(0).size();
        const auto outRank = function->get_results().at(0)->get_input_shape(0).size();
        if (inRank == 0 || outRank == 0) {
            throw LayerTestsUtils::VpuSkipTestException("SCALAR case is not supported by run-time");
        }
        if (inRank > 4 || outRank > 4) {
            throw LayerTestsUtils::VpuSkipTestException(">4D case is not supported by run-time");
        }
    }
};

class VPUXSqueezeUnsqueezeLayerTest_VPU3700 : public VPUXSqueezeUnsqueezeLayerTest {
    void SkipBeforeLoad() override {
        VPUXSqueezeUnsqueezeLayerTest::SkipBeforeLoad();
        // Tracking number [E#85137]
        if (getBackendName(*getCore()) == "LEVEL0") {
            throw LayerTestsUtils::VpuSkipTestException("Level0: failure on device");
        }
    }
};

class VPUXSqueezeUnsqueezeLayerTest_VPU3720 : public VPUXSqueezeUnsqueezeLayerTest {};

TEST_P(VPUXSqueezeUnsqueezeLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXSqueezeUnsqueezeLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

std::map<std::vector<size_t>, std::vector<std::vector<int>>> axesVectors = {
        {{1, 1, 1, 1},
         {{-1},
          {0},
          {1},
          {2},
          {3},
          {0, 1},
          {0, 2},
          {0, 3},
          {1, 2},
          {2, 3},
          {0, 1, 2},
          {0, 2, 3},
          {1, 2, 3},
          {0, 1, 2, 3}}},
        {{1, 2, 3, 4}, {{0}}},
        {{2, 1, 3, 4}, {{1}}},
        {{1}, {{-1}, {0}}},
        {{1, 2}, {{0}}},
        {{2, 1}, {{1}, {-1}}},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<ngraph::helpers::SqueezeOpType> opTypes = {ngraph::helpers::SqueezeOpType::SQUEEZE,
                                                             ngraph::helpers::SqueezeOpType::UNSQUEEZE};
const auto paramConfig = testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(axesVectors)), ::testing::ValuesIn(opTypes),
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Basic, VPUXSqueezeUnsqueezeLayerTest_VPU3700, paramConfig,
                         SqueezeUnsqueezeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Basic, VPUXSqueezeUnsqueezeLayerTest_VPU3720, paramConfig,
                         SqueezeUnsqueezeLayerTest::getTestCaseName);

}  // namespace
