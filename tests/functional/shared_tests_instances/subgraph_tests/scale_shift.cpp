// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/subgraph/scaleshift.hpp>

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace SubgraphTestsDefinitions {

class KmbScaleShiftLayerTest: public ScaleShiftLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
};

TEST_P(KmbScaleShiftLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace SubgraphTestsDefinitions

using namespace SubgraphTestsDefinitions;

namespace {

std::vector<std::vector<std::vector<size_t>>> inShapes = {
    {{1, 8, 4, 4},     {1, 8, 1, 1}},
    {{1, 128, 32, 32}, {1, 128, 1, 1}},
    {{1, 512, 64, 64}, {1, 512, 1, 1}},
    {{1, 111, 3, 3},   {1, 111, 1, 1}},
};

std::vector<std::vector<float >> Scales = {
    {3.0f},
    {-3.0f}
};

std::vector<std::vector<float >> Shifts = {
    {3.0f},
    {-3.0f}
};

std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                         InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_CASE_P(smoke_scale_shift, KmbScaleShiftLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(inShapes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::ValuesIn(Scales),
                            ::testing::ValuesIn(Shifts)),
                        KmbScaleShiftLayerTest::getTestCaseName);

}  // namespace
