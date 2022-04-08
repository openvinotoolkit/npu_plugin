//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/gru_sequence.hpp"
#include <vector>
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXGRUSequenceLayerTest_VPU3720 : public GRUSequenceTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SetUp() override {
        inPrc = InferenceEngine::Precision::FP16;
        outPrc = InferenceEngine::Precision::FP16;
        GRUSequenceTest::SetUp();
    }
};

TEST_P(VPUXGRUSequenceLayerTest_VPU3720, CompareWithRefs_MLIR) {
    threshold = 0.06;
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using GRUDirection = ngraph::op::RecurrentSequenceDirection;

namespace {

const auto testMode = ngraph::helpers::SequenceTestsMode::PURE_SEQ;
const std::vector<size_t> seqLength{1, 5};
const size_t batchSize = 2;
const size_t hiddenSize = 4;
const std::vector<std::string> activations = {"sigmoid", "tanh"};
const float clip = 0.0f;
const std::vector<bool> shouldLinearBeforeReset{true};
const std::vector<GRUDirection> directionMode{GRUDirection::FORWARD};
const InferenceEngine::Precision netPrecisions = InferenceEngine::Precision::FP16;

INSTANTIATE_TEST_SUITE_P(smoke_GRUSequence_VPU3720, VPUXGRUSequenceLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(testMode), ::testing::ValuesIn(seqLength),
                                            ::testing::Values(batchSize), ::testing::Values(hiddenSize),
                                            ::testing::Values(activations), ::testing::Values(clip),
                                            ::testing::ValuesIn(shouldLinearBeforeReset),
                                            ::testing::ValuesIn(directionMode), ::testing::Values(netPrecisions),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         GRUSequenceTest::getTestCaseName);
}  // namespace
