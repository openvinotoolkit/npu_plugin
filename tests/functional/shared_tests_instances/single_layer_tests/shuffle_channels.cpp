// Copyright (C) 2018-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "single_layer_tests/shuffle_channels.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {
class ShuffleChannelsLayerTestCommon :
        public ShuffleChannelsLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class ShuffleChannelsLayerTest_NPU3700 : public ShuffleChannelsLayerTestCommon {};
class ShuffleChannelsLayerTest_NPU3720 : public ShuffleChannelsLayerTestCommon {};

TEST_P(ShuffleChannelsLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(ShuffleChannelsLayerTest_NPU3720, SW) {
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

const auto params0 =
        testing::Combine(testing::ValuesIn(shuffleParameters), testing::ValuesIn(netPrecisions),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
                         testing::ValuesIn(inputShapes), testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_CASE_P(smoke_ShuffleChannels, ShuffleChannelsLayerTest_NPU3700, params0,
                        ShuffleChannelsLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ShuffleChannels, ShuffleChannelsLayerTest_NPU3720, params0,
                        ShuffleChannelsLayerTest_NPU3720::getTestCaseName);

}  // namespace

namespace {  // conformance scenarios

const std::vector<std::vector<size_t>> inShapes = {
        {1, 116, 28, 28}, {1, 232, 14, 14}, {1, 464, 7, 7},  {1, 32, 28, 28}, {1, 64, 14, 14},
        {1, 128, 7, 7},   {1, 24, 28, 28},  {1, 48, 14, 14}, {1, 96, 7, 7},
};

const std::vector<std::tuple<int, int>> shParams = {
        std::make_tuple(1, 2)  // axis=1, group=2
};

const auto params1 =
        testing::Combine(testing::ValuesIn(shParams), testing::Values(InferenceEngine::Precision::FP16),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                         testing::Values(InferenceEngine::Layout::ANY), testing::Values(InferenceEngine::Layout::ANY),
                         testing::ValuesIn(inShapes), testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto precommit_params = testing::Combine(
        testing::ValuesIn(shParams), testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED), testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY), testing::Values(std::vector<size_t>{1, 4, 3, 2}),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// --------- NPU3700 ---------

INSTANTIATE_TEST_CASE_P(conform_ShuffleChannels, ShuffleChannelsLayerTest_NPU3700, params1,
                        ShuffleChannelsLayerTest_NPU3700::getTestCaseName);

// --------- NPU3720 ---------

INSTANTIATE_TEST_CASE_P(conform_ShuffleChannels, ShuffleChannelsLayerTest_NPU3720, params1,
                        ShuffleChannelsLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(conform_precommit_ShuffleChannels, ShuffleChannelsLayerTest_NPU3720, precommit_params,
                        ShuffleChannelsLayerTest_NPU3720::getTestCaseName);

}  // namespace
