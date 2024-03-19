// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include <common/functions.h>
#include "single_layer_tests/gather_elements.hpp"
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {
class GatherElementsLayerTestCommon :
        public GatherElementsLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {};

class GatherElementsLayerTest_NPU3700 : public GatherElementsLayerTestCommon {
    void SkipBeforeInfer() override {
        if (getBackendName(*getCore()) == "LEVEL0") {
            throw LayerTestsUtils::VpuSkipTestException("Bad results on Level0");
        }
    }
};

class GatherElementsLayerTest_NPU3720 : public GatherElementsLayerTestCommon {};

TEST_P(GatherElementsLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(GatherElementsLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> dPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<InferenceEngine::Precision> iPrecisions = {InferenceEngine::Precision::I32};

const std::vector<int> axes_set1 = {-1, 0, 1};
const std::vector<int> axes_set2 = {-2, 1};
const std::vector<int> axes_set3 = {0};

// ------ NPU3700 ------
INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_set1, GatherElementsLayerTest_NPU3700,
                         testing::Combine(testing::Values(std::vector<size_t>{2, 2}),
                                          testing::Values(std::vector<size_t>{2, 2}), testing::ValuesIn(axes_set1),
                                          testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         GatherElementsLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_set2, GatherElementsLayerTest_NPU3700,
                         testing::Combine(testing::Values(std::vector<size_t>{5, 7, 9, 1}),
                                          testing::Values(std::vector<size_t>{5, 7, 9, 1}),
                                          testing::ValuesIn(axes_set2), testing::ValuesIn(dPrecisions),
                                          testing::ValuesIn(iPrecisions),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         GatherElementsLayerTest_NPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GatherElements_set3, GatherElementsLayerTest_NPU3700,
                         testing::Combine(testing::Values(std::vector<size_t>{2, 2, 1}),
                                          testing::Values(std::vector<size_t>{4, 2, 1}), testing::ValuesIn(axes_set3),
                                          testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions),
                                          testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         GatherElementsLayerTest_NPU3700::getTestCaseName);

// ------ NPU3720 ------
const auto GatherElements_PRECOMMIT_set1 =
        ::testing::Combine(testing::Values(std::vector<size_t>{2, 2}), testing::Values(std::vector<size_t>{2, 2}),
                           testing::ValuesIn(axes_set1), testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions),
                           testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto GatherElements_PRECOMMIT_set2 = ::testing::Combine(
        testing::Values(std::vector<size_t>{5, 7, 9, 1}), testing::Values(std::vector<size_t>{5, 7, 9, 1}),
        testing::ValuesIn(axes_set2), testing::ValuesIn(dPrecisions), testing::ValuesIn(iPrecisions),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

const auto GatherElements_PRECOMMIT_set3 = ::testing::Combine(
        ::testing::Values(std::vector<size_t>{2, 2, 1}), ::testing::Values(std::vector<size_t>{4, 2, 1}),
        ::testing::ValuesIn(axes_set3), ::testing::ValuesIn(dPrecisions), ::testing::ValuesIn(iPrecisions),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// ------ NPU3720 ------
INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set1, GatherElementsLayerTest_NPU3720,
                         GatherElements_PRECOMMIT_set1, GatherElementsLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set2, GatherElementsLayerTest_NPU3720,
                         GatherElements_PRECOMMIT_set2, GatherElementsLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_GatherElements_set3, GatherElementsLayerTest_NPU3720,
                         GatherElements_PRECOMMIT_set3, GatherElementsLayerTest_NPU3720::getTestCaseName);

}  // namespace
