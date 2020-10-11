// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reshape.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbReshapeLayerTest : public ReshapeLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

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
// [NOT_FOUND] DYN_BATCH_ENABLED key is not supported for VPU
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
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                            ::testing::Values(std::map<std::string, std::string>({{CONFIG_KEY(DYN_BATCH_ENABLED), CONFIG_VALUE(YES)}}))),
                        ReshapeLayerTest::getTestCaseName);

// Test fails with message:
// C++ exception with description "Size of dims(3) and format(NHWC) are inconsistent.
// It is possible to pass the test by setting data-member outLayout = InferenceEngine::Layout::ANY
// in constructor KmbLayerTestsCommon::KmbLayerTestsCommon().
// Please see file kmb-plugin/tests/functional/shared_tests_instances/kmb_layer_test.cpp:28
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_ReshapeCheck, KmbReshapeLayerTest,
                        ::testing::Combine(
                            ::testing::Values(true),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
                            ::testing::Values(std::vector<size_t>({10, 0, 100})),
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                            ::testing::Values(std::map<std::string, std::string>({}))),
                        ReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ReshapeCheck4Dto4DTensor, KmbReshapeLayerTest,
                        ::testing::Combine(
                            ::testing::Values(true),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(std::vector<size_t>({1, 1, 1, 1000})),
                            ::testing::Values(std::vector<size_t>({1, 1000, 1, 1})),
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                            ::testing::Values(std::map<std::string, std::string>({}))),
                        ReshapeLayerTest::getTestCaseName);

}  // namespace
