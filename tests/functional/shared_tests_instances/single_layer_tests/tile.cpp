// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "kmb_layer_test.hpp"
#include "single_layer_tests/tile.hpp"

namespace LayerTestsDefinitions {

class KmbTileLayerTest : public TileLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        if (isCompilerMCM()) {
            // 1. Test fails with errors like this:
            // C++ exception with description "Size of dims(3) and format(NHWC) are inconsistent.
            // openvino/inference-engine/src/inference_engine/ie_layouts.cpp:138" thrown in the test body.
            // [Track number: S#40116]
            // 2. Test fails with errors like this:
            // C++ exception with description "Size of dims(6) and format(NHWC) are inconsistent.
            // openvino/inference-engine/src/inference_engine/ie_layouts.cpp:138" thrown in the test body.
            // [Track number: S#40116]
            // 3. Test fails with error:
            // C++ exception with description "Tile operation has a form that is not supported.
            // Tile_2 should be converted to TileIE operation.
            // openvino/inference-engine/src/legacy_api/src/convert_function_to_cnn_network.cpp:663
            // openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
            // [Track number: S#40116]
            throw LayerTestsUtils::KmbSkipTestException("Issues with MCM compiler");
        }
    }
};

TEST_P(KmbTileLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<std::vector<int64_t>> repeats = {
        // tile by single axes
        {1, 1, 1, 5},
        {1, 1, 5},
        {1, 5, 1},
        {5, 1, 1},
        {1, 8},
        {8, 1},

        // tile by multiple axes
        {1, 2, 3},
        {2, 3, 1},
        {3, 1, 2},

        // identical tile case
        {1, 1, 1},

        // input shapes with more than 4D is not supported by runtime yet
        // {1, 1, 1, 2, 1, 2}

        // looks like this values is too big. Test fails due result missmatch between CPU an KMB
        // {1, 1, 1, 128}, {1, 1, 128, 1}, {1, 128, 1, 1}, {128, 1, 1, 1},
};

const std::vector<std::vector<size_t>> inputShapes = {
        {2},
        {2, 3},

        {3, 4, 2},
        {2, 3, 4, 2},
        {1, 1, 128, 1},

        // input shapes with more than 4D is not supported by runtime yet
        // {1, 4, 3, 1, 3, 1}
};

INSTANTIATE_TEST_SUITE_P(Tile, KmbTileLayerTest,
                        ::testing::Combine(::testing::ValuesIn(repeats), ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbTileLayerTest::getTestCaseName);
}  // namespace
