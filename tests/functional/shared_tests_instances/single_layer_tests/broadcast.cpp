
// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/broadcast.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbBroadcastLayerTest : public BroadcastLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbBroadcastLayerTest, BroadcastCheck_MLIR) {
    useCompilerMLIR();
    Run();
}

} // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
// Common params

const std::vector<InferenceEngine::Precision> inputPrecision = {InferenceEngine::Precision::FP32};

std::vector<std::vector<size_t>> inShapesNumpy = {{3, 1}};
std::vector<std::vector<size_t>> targetShapesNumpy = {{2, 3, 6}};

INSTANTIATE_TEST_CASE_P(smoke_BroadcastCheck, KmbBroadcastLayerTest,
                        ::testing::Combine(
                                           ::testing::Values(targetShapesNumpy[0]),
                                           ::testing::Values(ngraph::AxisSet{}),
                                           ::testing::Values(ngraph::op::BroadcastType::NUMPY),
                                           ::testing::Values(inShapesNumpy[0]),
                                           ::testing::ValuesIn(inputPrecision),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbBroadcastLayerTest::getTestCaseName);

}  // namespace