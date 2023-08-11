// Copyright (C) 2019-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reshape.hpp"

#include <vector>

#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {
class VPUXReshapeLayerTest : public ReshapeLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class VPUXReshapeLayerTest_VPU3700 : public VPUXReshapeLayerTest {
    void ConfigureNetwork() override {
        LayerTestsUtils::KmbLayerTestsCommon::ConfigureNetwork();
    }

    void SkipBeforeInfer() override {
    }
};

class VPUXReshapeLayerTest_VPU3720 : public VPUXReshapeLayerTest {};

TEST_P(VPUXReshapeLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXReshapeLayerTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse1, VPUXReshapeLayerTest_VPU3700,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 1, 1, 100}),
                                                              std::vector<size_t>({1, 100, 1, 1})),
                                            ::testing::Values(std::vector<int64_t>({0, 100})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         VPUXReshapeLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse2, VPUXReshapeLayerTest_VPU3700,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 2, 10, 10})),
                                            ::testing::Values(std::vector<int64_t>({1, 0, 100})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         VPUXReshapeLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand1, VPUXReshapeLayerTest_VPU3700,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 100})),
                                            ::testing::Values(std::vector<int64_t>({0, 100, 1, 1})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         VPUXReshapeLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand2, VPUXReshapeLayerTest_VPU3700,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 2, 100})),
                                            ::testing::Values(std::vector<int64_t>({0, 0, 10, 10})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         VPUXReshapeLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand3, VPUXReshapeLayerTest_VPU3700,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>{4}),
                                            ::testing::Values(std::vector<int64_t>{1, 1, 2, 2}),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         VPUXReshapeLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeGeneric1, VPUXReshapeLayerTest_VPU3700,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 1, 1, 1000})),
                                            ::testing::Values(std::vector<int64_t>({1, 1000, 1, 1})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         VPUXReshapeLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeGeneric2, VPUXReshapeLayerTest_VPU3700,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>{1, 4, 2, 2}),
                                            ::testing::Values(std::vector<int64_t>{1, 2, 4, 2}),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         VPUXReshapeLayerTest_VPU3700::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse1, VPUXReshapeLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 1, 1, 100}),
                                                              std::vector<size_t>({1, 100, 1, 1})),
                                            ::testing::Values(std::vector<int64_t>({0, 100})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         VPUXReshapeLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse2, VPUXReshapeLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 2, 10, 10})),
                                            ::testing::Values(std::vector<int64_t>({1, 0, 100})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         VPUXReshapeLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand1, VPUXReshapeLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 100})),
                                            ::testing::Values(std::vector<int64_t>({0, 100, 1, 1})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         VPUXReshapeLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand2, VPUXReshapeLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 2, 100})),
                                            ::testing::Values(std::vector<int64_t>({0, 0, 10, 10})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         VPUXReshapeLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand3, VPUXReshapeLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>{4}),
                                            ::testing::Values(std::vector<int64_t>{1, 1, 2, 2}),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         VPUXReshapeLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeGeneric1, VPUXReshapeLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 1, 1, 1000})),
                                            ::testing::Values(std::vector<int64_t>({1, 1000, 1, 1})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         VPUXReshapeLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ReshapeGeneric2, VPUXReshapeLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>{1, 4, 2, 2}),
                                            ::testing::Values(std::vector<int64_t>{1, 2, 4, 2}),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         VPUXReshapeLayerTest_VPU3720::getTestCaseName);

}  // namespace
