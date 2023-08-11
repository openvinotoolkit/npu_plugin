//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/squeeze_unsqueeze.hpp"
#include <vector>
#include "common/functions.h"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXSqueezeUnsqueezeLayerTest_VPU3700 :
        public SqueezeUnsqueezeLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        const auto inRank = function->get_parameters().at(0)->get_output_shape(0).size();
        const auto outRank = function->get_results().at(0)->get_input_shape(0).size();

        if (inRank == 0 || outRank == 0) {
            throw LayerTestsUtils::KmbSkipTestException("SCALAR case is not supported by run-time");
        }
        if (inRank > 4 || outRank > 4) {
            throw LayerTestsUtils::KmbSkipTestException(">4D case is not supported by run-time");
        }

        // [Track number: #E20158]
        if (getBackendName(*getCore()) == "LEVEL0") {
            throw LayerTestsUtils::KmbSkipTestException("Level0: failure on device");
        }
    }
};

TEST_P(VPUXSqueezeUnsqueezeLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
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

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_Basic, VPUXSqueezeUnsqueezeLayerTest_VPU3700,
                         ::testing::Combine(::testing::ValuesIn(CommonTestUtils::combineParams(axesVectors)),
                                            ::testing::ValuesIn(opTypes), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         SqueezeUnsqueezeLayerTest::getTestCaseName);

}  // namespace
