//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "single_layer_tests/reverse_sequence.hpp"

namespace LayerTestsDefinitions {

class VPUXReverseSequenceLayerTest :
        public ReverseSequenceLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXReverseSequenceLayerTest_VPU3700 : public VPUXReverseSequenceLayerTest {};
class VPUXReverseSequenceLayerTest_VPU3720 : public VPUXReverseSequenceLayerTest {};

TEST_P(VPUXReverseSequenceLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}
TEST_P(VPUXReverseSequenceLayerTest_VPU3720, HW) {
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

const std::vector<std::vector<size_t>> inputShapesVPU3720 = {{3, 10}, {3, 10, 12}, {3, 10, 11, 20}};

const std::vector<std::vector<size_t>> reversSeqLengthsVecShapes = {{3}};

const std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {ngraph::helpers::InputLayerType::CONSTANT,
                                                                          ngraph::helpers::InputLayerType::PARAMETER};

INSTANTIATE_TEST_SUITE_P(Basic_smoke, VPUXReverseSequenceLayerTest_VPU3700,
                         ::testing::Combine(::testing::ValuesIn(batchAxisIndices), ::testing::ValuesIn(seqAxisIndices),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(reversSeqLengthsVecShapes),
                                            ::testing::ValuesIn(secondaryInputTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXReverseSequenceLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReverseSequence_VPU3720, VPUXReverseSequenceLayerTest_VPU3720,
                         ::testing::Combine(::testing::ValuesIn(batchAxisIndices), ::testing::ValuesIn(seqAxisIndices),
                                            ::testing::ValuesIn(inputShapesVPU3720),
                                            ::testing::ValuesIn(reversSeqLengthsVecShapes),
                                            ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXReverseSequenceLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ReverseSequence_VPU3720, VPUXReverseSequenceLayerTest_VPU3720,
                         ::testing::Combine(::testing::ValuesIn(batchAxisIndices), ::testing::ValuesIn(seqAxisIndices),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(reversSeqLengthsVecShapes),
                                            ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                         VPUXReverseSequenceLayerTest_VPU3720::getTestCaseName);

}  // namespace
