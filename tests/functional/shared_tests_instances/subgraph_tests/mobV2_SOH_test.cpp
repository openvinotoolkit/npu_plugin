//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "shared_test_classes/subgraph/mobV2_SOH.hpp"
#include <vector>
#include "vpu_ov1_layer_test.hpp"

namespace SubgraphTestsDefinitions {
class VPUXMobilenetV2SlicedTest : public mobilenetV2SlicedTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    /* tests for mobilenet v2 split over H unequal subtensors
            input
              |
            groupConv
              |
            Add1
              |
            Clamp
              |
            Conv
              |
            Add2
              |
            output
    */
};

TEST_P(VPUXMobilenetV2SlicedTest, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
};

}  // namespace SubgraphTestsDefinitions

using namespace SubgraphTestsDefinitions;
using namespace ngraph::helpers;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{{"LOG_LEVEL", "LOG_INFO"}}};

INSTANTIATE_TEST_CASE_P(smoke_mobilenetV2SlicedTest, VPUXMobilenetV2SlicedTest,
                        ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()),
                                           ::testing::ValuesIn(configs)),
                        mobilenetV2SlicedTest::getTestCaseName);
}  // namespace
