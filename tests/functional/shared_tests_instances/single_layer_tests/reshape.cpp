// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reshape.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbReshapeLayerTest : public ReshapeLayerTestRevise, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void ConfigureNetwork() override {
        LayerTestsUtils::KmbLayerTestsCommon::ConfigureNetwork();
        if (isCompilerMCM()) {
            disableMcmPasses({{"kmb_adapt", "ConvertToImplicitOps"}, {"adapt", "RemoveOps"}});
        }
    }

    void SkipBeforeInfer() override {
        if (isCompilerMCM()) {
            InferenceEngine::SizeVector inputShape{};
            std::tie(std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore,
                inputShape, std::ignore, std::ignore, std::ignore) = GetParam();

            if (inputShape.size() == 4 && inputShape[0] != 1) {
                throw LayerTestsUtils::KmbSkipTestException("Runtime does not support Reshape with input tensor batch != 1");
            }
        }
    }
};

TEST_P(KmbReshapeLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbReshapeLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
};

INSTANTIATE_TEST_CASE_P(smoke_ReshapeCollapse1, KmbReshapeLayerTest,
    ::testing::Combine(
        ::testing::Values(true),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 1, 1, 100}), std::vector<size_t>({1, 100, 1, 1})),
        ::testing::Values(std::vector<int64_t>({0, 100})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(std::map<std::string, std::string>({}))),
    KmbReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ReshapeCollapse2, KmbReshapeLayerTest,
    ::testing::Combine(
        ::testing::Values(true),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 2, 10, 10})),
        ::testing::Values(std::vector<int64_t>({1, 0, 100})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(std::map<std::string, std::string>({}))),
    KmbReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ReshapeExpand1, KmbReshapeLayerTest,
    ::testing::Combine(
        ::testing::Values(true),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 100})),
        ::testing::Values(std::vector<int64_t>({0, 100, 1, 1})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(std::map<std::string, std::string>({}))),
    KmbReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ReshapeExpand2, KmbReshapeLayerTest,
    ::testing::Combine(
        ::testing::Values(true),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 2, 100})),
        ::testing::Values(std::vector<int64_t>({0, 0, 10, 10})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(std::map<std::string, std::string>({}))),
    KmbReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ReshapeExpand3, KmbReshapeLayerTest,
    ::testing::Combine(
        ::testing::Values(true),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{4}),
        ::testing::Values(std::vector<int64_t>{1, 1, 2, 2}),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(std::map<std::string, std::string>({}))),
    KmbReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ReshapeGeneric1, KmbReshapeLayerTest,
    ::testing::Combine(
        ::testing::Values(true),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 1, 1, 1000})),
        ::testing::Values(std::vector<int64_t>({1, 1000, 1, 1})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(std::map<std::string, std::string>({}))),
    KmbReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ReshapeGeneric2, KmbReshapeLayerTest,
    ::testing::Combine(
        ::testing::Values(true),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 4, 2, 2}),
        ::testing::Values(std::vector<int64_t>{1, 2, 4, 2}),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(std::map<std::string, std::string>({}))),
    KmbReshapeLayerTest::getTestCaseName);

const std::vector<InferenceEngine::Precision> netPrecisions_pass_mcm = {
    InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_CASE_P(smoke_ReshapeCheck_pass_mcm, KmbReshapeLayerTest,
                        ::testing::Combine(
                            ::testing::Values(true),
                            ::testing::ValuesIn(netPrecisions_pass_mcm),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
                            ::testing::Values(std::vector<int64_t>({10, 0, 100})),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::Values(std::map<std::string, std::string>({}))),
                        KmbReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ReshapeCheck4Dto4DTensor_pass_mcm, KmbReshapeLayerTest,
                        ::testing::Combine(
                            ::testing::Values(true),
                            ::testing::ValuesIn(netPrecisions_pass_mcm),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(std::vector<size_t>({1, 1, 1, 1000})),
                            ::testing::Values(std::vector<int64_t>({1, 1000, 1, 1})),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::Values(std::map<std::string, std::string>({}))),
                        KmbReshapeLayerTest::getTestCaseName);

}  // namespace
