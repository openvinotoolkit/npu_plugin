// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reshape.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbReshapeLayerTest : public ReshapeLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void ConfigureNetwork() override {
        LayerTestsUtils::KmbLayerTestsCommon::ConfigureNetwork();
    }

    void SkipBeforeInfer() override {
    }
};

TEST_P(KmbReshapeLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbReshapeLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

class KmbReshapeLayerTest_VPU3720 : public ReshapeLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbReshapeLayerTest_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse1, KmbReshapeLayerTest,
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
                         KmbReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse2, KmbReshapeLayerTest,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 2, 10, 10})),
                                            ::testing::Values(std::vector<int64_t>({1, 0, 100})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         KmbReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand1, KmbReshapeLayerTest,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 100})),
                                            ::testing::Values(std::vector<int64_t>({0, 100, 1, 1})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         KmbReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand2, KmbReshapeLayerTest,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 2, 100})),
                                            ::testing::Values(std::vector<int64_t>({0, 0, 10, 10})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         KmbReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand3, KmbReshapeLayerTest,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>{4}),
                                            ::testing::Values(std::vector<int64_t>{1, 1, 2, 2}),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         KmbReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeGeneric1, KmbReshapeLayerTest,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 1, 1, 1000})),
                                            ::testing::Values(std::vector<int64_t>({1, 1000, 1, 1})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         KmbReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeGeneric2, KmbReshapeLayerTest,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>{1, 4, 2, 2}),
                                            ::testing::Values(std::vector<int64_t>{1, 2, 4, 2}),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         KmbReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse1, KmbReshapeLayerTest_VPU3720,
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
                         KmbReshapeLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeCollapse2, KmbReshapeLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 2, 10, 10})),
                                            ::testing::Values(std::vector<int64_t>({1, 0, 100})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         KmbReshapeLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand1, KmbReshapeLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 100})),
                                            ::testing::Values(std::vector<int64_t>({0, 100, 1, 1})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         KmbReshapeLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand2, KmbReshapeLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 2, 100})),
                                            ::testing::Values(std::vector<int64_t>({0, 0, 10, 10})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         KmbReshapeLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeExpand3, KmbReshapeLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>{4}),
                                            ::testing::Values(std::vector<int64_t>{1, 1, 2, 2}),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         KmbReshapeLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReshapeGeneric1, KmbReshapeLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({1, 1, 1, 1000})),
                                            ::testing::Values(std::vector<int64_t>({1, 1000, 1, 1})),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         KmbReshapeLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_ReshapeGeneric2, KmbReshapeLayerTest_VPU3720,
                         ::testing::Combine(::testing::Values(true), ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>{1, 4, 2, 2}),
                                            ::testing::Values(std::vector<int64_t>{1, 2, 4, 2}),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                            ::testing::Values(std::map<std::string, std::string>({}))),
                         KmbReshapeLayerTest_VPU3720::getTestCaseName);

}  // namespace
