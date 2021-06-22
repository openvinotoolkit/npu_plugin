// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/opsets/opset3.hpp>

#include "single_layer_tests/depth_to_space.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class KmbDepthToSpaceLayerTest: public DepthToSpaceLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
        void SkipBeforeLoad() override {
            if (envConfig.IE_KMB_TESTS_RUN_INFER) {
                throw LayerTestsUtils::KmbSkipTestException("layer test networks hang the board");
            }
        }
    };

    TEST_P(KmbDepthToSpaceLayerTest, CompareWithRefs) {
        Run();
    }
}  // namespace LayerTestsDefinitions


using namespace LayerTestsDefinitions;
using namespace ngraph::opset3;

namespace {
    const std::vector<InferenceEngine::Precision> inputPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::U8,
            InferenceEngine::Precision::FP16, // CPU-plugin has parameter I16, but KMB does not
    };                                        // support it. So I16 is changed to FP16.

    const std::vector<DepthToSpace::DepthToSpaceMode> modes = {
            DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
            DepthToSpace::DepthToSpaceMode::DEPTH_FIRST};

    const std::vector<std::vector<size_t >> inputShapesBS2 = {
            {1, 4, 1, 1}, {1, 4, 2, 2}, {1, 4, 3, 3}, {2, 32, 3, 3}, {2, 16, 5, 4}};
            // {1, 8, 1, 1, 1}, {1, 8, 2, 2, 2}, {1, 8, 3, 3, 3}, {2, 32, 3, 3, 3}, {2, 16, 5, 4, 6}};
            // These 5-dimensional values from CPU-test, but kmb-plugin does not support dims.size() > 4.
            // Therefore they are commented.
            // For details please see: kmb-plugin/src/utils/dims_parser.cpp

    const auto DepthToSpaceBS2 = ::testing::Combine(
            ::testing::ValuesIn(inputShapesBS2),
            ::testing::ValuesIn(inputPrecisions),
            ::testing::ValuesIn(modes),
            ::testing::Values(2),
            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpaceBS2, KmbDepthToSpaceLayerTest,
            DepthToSpaceBS2, KmbDepthToSpaceLayerTest::getTestCaseName);

    const std::vector<std::vector<size_t >> inputShapesBS3 = {
            {1, 9, 1, 1}, {1, 9, 2, 2}, {1, 9, 3, 3}, {2, 36, 3, 3}, {2, 27, 5, 4}};
            // {1, 27, 1, 1, 1}, {1, 27, 2, 2, 2}, {1, 27, 3, 3, 3}, {2, 108, 3, 3, 3}, {2, 54, 5, 4, 6}};
            // These 5-dimensional values from CPU-test, but kmb-plugin does not support dims.size() > 4.
            // Therefore they are commented.
            // For details please see: kmb-plugin/src/utils/dims_parser.cpp

    const auto DepthToSpaceBS3 = ::testing::Combine(
            ::testing::ValuesIn(inputShapesBS3),
            ::testing::ValuesIn(inputPrecisions),
            ::testing::ValuesIn(modes),
            ::testing::Values(3),
            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    INSTANTIATE_TEST_SUITE_P(smoke_DepthToSpaceBS3, KmbDepthToSpaceLayerTest,
            DepthToSpaceBS3, KmbDepthToSpaceLayerTest::getTestCaseName);

}  // namespace
