// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/tile.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbTileLayerTest : public TileLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbTileLayerTest, TileCheck) {
    Run();
}

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
};

const std::vector<std::vector<int64_t>> repeats = {
    {1, 2, 3},
    {2, 1, 1},
    {2, 3, 1},
    {2, 2, 2},
};

// Test fails with errors like this:
// C++ exception with description "Size of dims(3) and format(NHWC) are inconsistent.
// openvino/inference-engine/src/inference_engine/ie_layouts.cpp:138" thrown in the test body.
// [Track number: S#40116]
INSTANTIATE_TEST_CASE_P(DISABLED_Tile, KmbTileLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(repeats),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(std::vector<size_t>({2, 3, 4})),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbTileLayerTest::getTestCaseName);

// Test fails with errors like this:
// C++ exception with description "Size of dims(6) and format(NHWC) are inconsistent.
// openvino/inference-engine/src/inference_engine/ie_layouts.cpp:138" thrown in the test body.
// [Track number: S#40116]
INSTANTIATE_TEST_CASE_P(DISABLED_Tile6d, KmbTileLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<int64_t>({1, 1, 1, 2, 1, 2})),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(std::vector<size_t>({1, 4, 3, 1, 3, 1})),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbTileLayerTest::getTestCaseName);

// Test fails with error:
// C++ exception with description "Tile operation has a form that is not supported.
// Tile_2 should be converted to TileIE operation.
// openvino/inference-engine/src/legacy_api/src/convert_function_to_cnn_network.cpp:663
// openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
// [Track number: S#40116]
INSTANTIATE_TEST_CASE_P(DISABLED_Kmb_Specific_Tile, KmbTileLayerTest,
                        ::testing::Combine(
                            ::testing::Values(std::vector<int64_t>({1, 1, 88})),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(std::vector<size_t>({1, 1, 128, 1})),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbTileLayerTest::getTestCaseName);

}  // namespace
