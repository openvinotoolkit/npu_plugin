//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/grn.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXGRNLayerTest : public GrnLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXGRNLayerTest_VPU3700 : public VPUXGRNLayerTest {};

TEST_P(VPUXGRNLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,  // Testing FP32/FP16 netPrecision functionality only for small scope of
        InferenceEngine::Precision::FP16   // tests: VPUXGRNLayerTest, VPUXSplitLayerTest, VPUXCTCGreedyDecoderLayerTest
};

const std::vector<InferenceEngine::SizeVector> inShapes = {
        InferenceEngine::SizeVector{1, 3, 30, 30},
        InferenceEngine::SizeVector{1, 24, 128, 224},
};

const std::vector<float> biases = {
        0.33f,
        1.1f,
};

const auto params = testing::Combine(
        testing::ValuesIn(netPrecisions), testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::Values(InferenceEngine::Layout::NCHW),
        testing::Values(InferenceEngine::Layout::ANY), testing::ValuesIn(inShapes), testing::ValuesIn(biases),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(smoke_GRN_test, VPUXGRNLayerTest_VPU3700, params, GrnLayerTest::getTestCaseName);

}  // namespace
