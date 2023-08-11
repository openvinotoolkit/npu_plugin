// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "kmb_layer_test.hpp"
#include "single_layer_tests/shuffle_channels.hpp"

namespace LayerTestsDefinitions {
class VPUXShuffleChannelsLayerTest :
        public ShuffleChannelsLayerTest,
        virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXShuffleChannelsLayerTest_VPU3700 : public VPUXShuffleChannelsLayerTest {};
class VPUXShuffleChannelsLayerTest_VPU3720 : public VPUXShuffleChannelsLayerTest {};

TEST_P(VPUXShuffleChannelsLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}
TEST_P(VPUXShuffleChannelsLayerTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<std::vector<size_t>> inputShapes = {{3, 4, 9, 5}, {2, 16, 24, 15}, {1, 32, 12, 25}};

const std::vector<std::tuple<int, int>> shuffleParameters = {std::make_tuple(1, 2), std::make_tuple(-3, 2),
                                                             std::make_tuple(2, 3), std::make_tuple(-2, 3),
                                                             std::make_tuple(3, 5), std::make_tuple(-1, 5)};

const auto params =
        testing::Combine(testing::ValuesIn(shuffleParameters), testing::ValuesIn(netPrecisions),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
                         testing::ValuesIn(inputShapes), testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_smoke_ShuffleChannels, VPUXShuffleChannelsLayerTest_VPU3700, params,
                        VPUXShuffleChannelsLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ShuffleChannels, VPUXShuffleChannelsLayerTest_VPU3720, params,
                        VPUXShuffleChannelsLayerTest_VPU3720::getTestCaseName);
}  // namespace

namespace {  // conformance scenarios

const std::vector<std::vector<size_t>> inShapes = {
        {1, 116, 28, 28}, {1, 232, 14, 14}, {1, 464, 7, 7},  {1, 32, 28, 28}, {1, 64, 14, 14},
        {1, 128, 7, 7},   {1, 24, 28, 28},  {1, 48, 14, 14}, {1, 96, 7, 7},
};

const std::vector<std::tuple<int, int>> shParams = {
        std::make_tuple(1, 2)  // axis=1, group=2
};

const auto params2 =
        testing::Combine(testing::ValuesIn(shParams), testing::Values(InferenceEngine::Precision::FP16),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
                         testing::ValuesIn(inShapes), testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(DISABLED_TMP_conform_ShuffleChannels, VPUXShuffleChannelsLayerTest_VPU3700, params2,
                        VPUXShuffleChannelsLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(conform_ShuffleChannels, VPUXShuffleChannelsLayerTest_VPU3720, params2,
                        VPUXShuffleChannelsLayerTest_VPU3720::getTestCaseName);

const auto precommit_params = testing::Combine(
        testing::ValuesIn(shParams), testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY), testing::Values(std::vector<size_t>{1, 4, 3, 2}),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(conform_precommit_ShuffleChannels, VPUXShuffleChannelsLayerTest_VPU3720, precommit_params,
                        VPUXShuffleChannelsLayerTest_VPU3720::getTestCaseName);
}  // namespace
