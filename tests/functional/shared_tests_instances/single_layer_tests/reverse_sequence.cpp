//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/reverse_sequence.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

class ReverseSequenceLayerTestCommon :
        public ReverseSequenceLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class ReverseSequenceLayerTest_NPU3700 : public ReverseSequenceLayerTestCommon {};
class ReverseSequenceLayerTest_NPU3720 : public ReverseSequenceLayerTestCommon {};

TEST_P(ReverseSequenceLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(ReverseSequenceLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16,
                                                               InferenceEngine::Precision::U8};

const std::vector<int64_t> batchAxisIndices = {0L};

const std::vector<int64_t> seqAxisIndices = {1L};

const std::vector<std::vector<size_t>> inputShapes = {{3, 10}};  //, 10, 20

const std::vector<std::vector<size_t>> inputShapesNPU3720 = {{3, 10}, {3, 10, 12}, {3, 10, 11, 20}};

const std::vector<std::vector<size_t>> reversSeqLengthsVecShapes = {{3}};

const std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {ngraph::helpers::InputLayerType::CONSTANT,
                                                                          ngraph::helpers::InputLayerType::PARAMETER};

INSTANTIATE_TEST_SUITE_P(smoke_Basic, ReverseSequenceLayerTest_NPU3700,
                         ::testing::Combine(::testing::ValuesIn(batchAxisIndices), ::testing::ValuesIn(seqAxisIndices),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(reversSeqLengthsVecShapes),
                                            ::testing::ValuesIn(secondaryInputTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         ReverseSequenceLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReverseSequence, ReverseSequenceLayerTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(batchAxisIndices), ::testing::ValuesIn(seqAxisIndices),
                                            ::testing::ValuesIn(inputShapesNPU3720),
                                            ::testing::ValuesIn(reversSeqLengthsVecShapes),
                                            ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         ReverseSequenceLayerTest_NPU3720::getTestCaseName);

}  // namespace
