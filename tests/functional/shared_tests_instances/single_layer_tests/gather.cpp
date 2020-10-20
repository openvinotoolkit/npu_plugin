// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/gather.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbGatherLayerTest: public GatherLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbGatherLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inputShapes = {
        std::vector<size_t>{10, 20, 30, 40},
};

const std::vector<std::vector<int>> indices = {
        std::vector<int>{0, 3, 2, 1},
};
const std::vector<std::vector<size_t>> indicesShapes = {
        std::vector<size_t>{4},
        // std::vector<size_t>{2, 2}  //  Only 1D shape for indices is supported
};

const std::vector<int> axes = {0, 1, 2, 3, /*-1*/};  // Only positive axis value is supported

const auto params = testing::Combine(
        testing::ValuesIn(indices),
        testing::ValuesIn(indicesShapes),
        testing::ValuesIn(axes),
        testing::ValuesIn(inputShapes),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

// nGraph parser doesn't contain specific gather parser
// [Track number: S#40603]
INSTANTIATE_TEST_CASE_P(
        smoke_Gather,
        KmbGatherLayerTest,
        params,
        KmbGatherLayerTest::getTestCaseName
);

}  // namespace
