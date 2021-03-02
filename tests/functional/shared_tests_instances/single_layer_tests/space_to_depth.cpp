// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <ngraph/opsets/opset3.hpp>

#include "single_layer_tests/space_to_depth.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class KmbSpaceToDepthLayerTest: public SpaceToDepthLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    };

    TEST_P(KmbSpaceToDepthLayerTest, CompareWithRefs) {
        Run();
    }
}  // namespace LayerTestsDefinitions


using namespace LayerTestsDefinitions;
using namespace ngraph::opset3;

namespace {
    const std::vector<InferenceEngine::Precision> inputPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16, // value from CPU-plugin I16 is changed for FP16
            InferenceEngine::Precision::U8
    };

    const std::vector<SpaceToDepth::SpaceToDepthMode> modes = {
            SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,
            SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST};

    const std::vector<std::vector<size_t >> inputShapesBS2 = {
            {1, 1, 2, 2}, {1, 1, 4, 4}, {1, 1, 6, 6}, {2, 8, 6, 6}, {2, 4, 10, 8}};
            // {1, 1, 2, 2, 2}, {1, 1, 4, 4, 4}, {1, 1, 6, 6, 6}, {2, 8, 6, 6, 6}, {2, 4, 10, 8, 12}};
            // These 5-dimensional values from CPU-test, but kmb-plugin does not support dims.size() > 4.
            // Therefore they are commented.
            // For details please see: kmb-plugin/src/utils/dims_parser.cpp


    const auto SpaceToDepthBS2 = ::testing::Combine(
            ::testing::ValuesIn(inputShapesBS2),
            ::testing::ValuesIn(inputPrecisions),
            ::testing::ValuesIn(modes),
            ::testing::Values(2),
            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    // All tests fail with one common error:
    // Expected: executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration) doesn't
    // throw an exception.
    // Actual: it throws:Unsupported operation: SpaceToDepth_29964 with name SpaceToDepth_29980 with type
    // SpaceToDepth with C++ type N6ngraph2op2v012SpaceToDepthE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1575
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64
    // [Track number: S#45262]
    INSTANTIATE_TEST_CASE_P(DISABLED_smoke_SpaceToDepthBS2, KmbSpaceToDepthLayerTest, SpaceToDepthBS2, KmbSpaceToDepthLayerTest::getTestCaseName);

    const std::vector<std::vector<size_t >> inputShapesBS3 = {
            {1, 1, 3, 3}, {1, 1, 6, 6}, {1, 1, 9, 9}, {2, 4, 9, 9}, {2, 3, 15, 12}};
            // {1, 1, 3, 3, 3}, {1, 1, 6, 6, 6}, {1, 1, 9, 9, 9}, {2, 4, 9, 9, 9}, {2, 3, 15, 12, 18}};
            // These 5-dimensional values from CPU-test, but kmb-plugin does not support dims.size() > 4.
            // Therefore they are commented.
            // For details please see: kmb-plugin/src/utils/dims_parser.cpp

            const auto SpaceToDepthBS3 = ::testing::Combine(
            ::testing::ValuesIn(inputShapesBS3),
            ::testing::ValuesIn(inputPrecisions),
            ::testing::ValuesIn(modes),
            ::testing::Values(3),
            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    // All tests fail with one common error:
    // Expected: executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration) doesn't
    // throw an exception.
    // Actual: it throws:Unsupported operation: SpaceToDepth_29964 with name SpaceToDepth_29980 with type
    // SpaceToDepth with C++ type N6ngraph2op2v012SpaceToDepthE
    // kmb-plugin/src/frontend_mcm/src/ngraph_mcm_frontend/passes/convert_to_mcm_model.cpp:1575
    // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64
    // [Track number: S#45262]
    INSTANTIATE_TEST_CASE_P(DISABLED_smoke_SpaceToDepthBS3, KmbSpaceToDepthLayerTest, SpaceToDepthBS3, KmbSpaceToDepthLayerTest::getTestCaseName);

}  // namespace
