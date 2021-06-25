// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "kmb_layer_test.hpp"
#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/subgraph/mobV2_SOH.hpp"

namespace SubgraphTestsDefinitions {

class KmbMobilenetV2SlicedTest: public mobilenetV2SlicedTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
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

TEST_P(KmbMobilenetV2SlicedTest, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions

using namespace SubgraphTestsDefinitions;
using namespace ngraph::helpers;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"LOG_LEVEL", "LOG_INFO"}
    }
};

INSTANTIATE_TEST_CASE_P(smoke_mobilenetV2SlicedTest, KmbMobilenetV2SlicedTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::ValuesIn(configs)
                            ),
                        mobilenetV2SlicedTest::getTestCaseName);
}  // namespace
