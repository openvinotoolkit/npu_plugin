//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/ctc_greedy_decoder_seq_len.hpp"
#include <vector>
#include "vpu_ov1_layer_test.hpp"

#include <ngraph/op/util/op_types.hpp>

using namespace ngraph;
namespace LayerTestsDefinitions {

class VPUXCTCGreedyDecoderSeqLenLayerTest :
        public CTCGreedyDecoderSeqLenLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class VPUXCTCGreedyDecoderSeqLenLayerTest_VPU3700 : public VPUXCTCGreedyDecoderSeqLenLayerTest {
    void SkipBeforeInfer() override {
        throw LayerTestsUtils::VpuSkipTestException("differs from the reference");
    }
};

class VPUXCTCGreedyDecoderSeqLenLayerTest_VPU3720 : public VPUXCTCGreedyDecoderSeqLenLayerTest {};

TEST_P(VPUXCTCGreedyDecoderSeqLenLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXCTCGreedyDecoderSeqLenLayerTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> probPrecisions = {InferenceEngine::Precision::FP16};
const std::vector<InferenceEngine::Precision> idxPrecisions = {InferenceEngine::Precision::I32};

std::vector<bool> mergeRepeated{true, false};

const auto inputShape =
        std::vector<std::vector<size_t>>{{1, 1, 1}, {4, 80, 80}, {80, 4, 80}, {80, 80, 4}, {8, 20, 128}};

const auto sequenceLengths = std::vector<int>{1, 50, 100};

const auto blankIndexes = std::vector<int>{0, 50};

const auto params = testing::Combine(::testing::ValuesIn(inputShape), ::testing::ValuesIn(sequenceLengths),
                                     ::testing::ValuesIn(probPrecisions), ::testing::ValuesIn(idxPrecisions),
                                     ::testing::ValuesIn(blankIndexes), ::testing::ValuesIn(mergeRepeated),
                                     ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// VPU3700
INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_CTCGreedyDecoderSeqLenTests, VPUXCTCGreedyDecoderSeqLenLayerTest_VPU3700,
                         params, CTCGreedyDecoderSeqLenLayerTest::getTestCaseName);

// VPU3720
INSTANTIATE_TEST_SUITE_P(smoke_CTCGreedyDecoderSeqLenTests, VPUXCTCGreedyDecoderSeqLenLayerTest_VPU3720, params,
                         CTCGreedyDecoderSeqLenLayerTest::getTestCaseName);
