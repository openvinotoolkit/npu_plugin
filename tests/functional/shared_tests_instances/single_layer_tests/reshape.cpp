// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reshape.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbReshapeLayerTest : public ReshapeLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeValidate() override {
        throw LayerTestsUtils::KmbSkipTestException("comparison fails");
    }
};

TEST_P(KmbReshapeLayerTest, ReshapeCheck) {
    Run();
}

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

// Test fails with message:
// C++ exception with description "[NOT_FOUND] DYN_BATCH_ENABLED key is not supported for VPU
// openvino/inference-engine/src/vpu/common/src/parsed_config_base.cpp:44
// openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
// [Track number: S#41220]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_ReshapeCheckDynBatch, KmbReshapeLayerTest,
                        ::testing::Combine(
                            ::testing::Values(true),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),
                            ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::Values(std::map<std::string, std::string>({{CONFIG_KEY(DYN_BATCH_ENABLED), CONFIG_VALUE(YES)}}))),
                        ReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ReshapeCheck_pass_mcm, KmbReshapeLayerTest,
                        ::testing::Combine(
                            ::testing::Values(true),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
                            ::testing::Values(std::vector<size_t>({10, 0, 100})),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::Values(std::map<std::string, std::string>({}))),
                        ReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ReshapeCheck4Dto4DTensor_pass_mcm, KmbReshapeLayerTest,
                        ::testing::Combine(
                            ::testing::Values(true),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(std::vector<size_t>({1, 1, 1, 1000})),
                            ::testing::Values(std::vector<size_t>({1, 1000, 1, 1})),
                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                            ::testing::Values(std::map<std::string, std::string>({}))),
                        ReshapeLayerTest::getTestCaseName);

const std::vector<InferenceEngine::Precision> netPrecisionsNetworks = {
        InferenceEngine::Precision::FP32,
        // InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_CASE_P(smoke_ReshapeCheckNetworkValues_1, KmbReshapeLayerTest,
        ::testing::Combine(
        ::testing::Values(true),
        ::testing::ValuesIn(netPrecisionsNetworks),
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{4}),
        ::testing::Values(std::vector<size_t>{1, 1, 2, 2}),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(std::map<std::string, std::string>({}))),
        ReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ReshapeCheckNetworkValues_2, KmbReshapeLayerTest,
        ::testing::Combine(
        ::testing::Values(true),
        ::testing::ValuesIn(netPrecisionsNetworks),
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Precision::FP16),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>{1, 4, 2, 2}),
        ::testing::Values(std::vector<size_t>{1, 2, 4, 2}),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(std::map<std::string, std::string>({}))),
        ReshapeLayerTest::getTestCaseName);

}  // namespace
