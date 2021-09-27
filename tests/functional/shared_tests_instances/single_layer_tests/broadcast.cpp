
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

// Numpy

const std::vector<InferenceEngine::Precision> inputPrecision = {InferenceEngine::Precision::FP16,
                                                                InferenceEngine::Precision::FP32};

std::vector<std::vector<size_t>> inShapesNumpy = {{3, 1}};
std::vector<std::vector<size_t>> targetShapesNumpy = {{2, 3, 6}};

const auto numpyBroadcastParams = ::testing::Combine(
                                           ::testing::Values(targetShapesNumpy[0]),
                                           ::testing::Values(ngraph::AxisSet{}), //not used in numpy mode
                                           ::testing::Values(ngraph::op::BroadcastType::NUMPY),
                                           ::testing::Values(inShapesNumpy[0]),
                                           ::testing::ValuesIn(inputPrecision),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_NumpyBroadcastCheck,
                        KmbBroadcastLayerTest,
                        numpyBroadcastParams,
                        KmbBroadcastLayerTest::getTestCaseName);


// Bidirectional

std::vector<std::vector<size_t>> inShapesBidi = {
        {4, 1},
        {1, 4, 1},
        {4, 1, 1}
};

std::vector<std::vector<size_t>> targetShapesBidi = {
        {2, 1, 4},
        {1, 4, 4},
        {1, 1, 2, 2}
};

const auto bidirectionalBroadcastParams = ::testing::Combine(
        ::testing::Values(targetShapesBidi[0]),
        ::testing::Values(ngraph::AxisSet{}), //not used in bidirectional mode
        ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
        ::testing::Values(inShapesBidi[0]),
        ::testing::ValuesIn(inputPrecision),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck,
                        KmbBroadcastLayerTest,
                        bidirectionalBroadcastParams,
                        KmbBroadcastLayerTest::getTestCaseName);

const auto bidirectionalBroadcastParams1 = ::testing::Combine(
        ::testing::Values(targetShapesBidi[1]),
        ::testing::Values(ngraph::AxisSet{}), //not used in bidirectional mode
        ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
        ::testing::Values(inShapesBidi[1]),
        ::testing::ValuesIn(inputPrecision),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck1,
                        KmbBroadcastLayerTest,
                        bidirectionalBroadcastParams1,
                        KmbBroadcastLayerTest::getTestCaseName);

const auto bidirectionalBroadcastParams2 = ::testing::Combine(
        ::testing::Values(targetShapesBidi[2]),
        ::testing::Values(ngraph::AxisSet{}), //not used in bidirectional mode
        ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
        ::testing::Values(inShapesBidi[2]),
        ::testing::ValuesIn(inputPrecision),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck2,
                        KmbBroadcastLayerTest,
                        bidirectionalBroadcastParams2,
                        KmbBroadcastLayerTest::getTestCaseName);

// Explicit

std::vector<std::vector<size_t>> inShapesExplicit = {
        {3, 1},
        {2, 4}
};

std::vector<std::vector<size_t>> targetShapesExplicit = {
        {2, 3, 1},
        {2, 3, 4}
};

std::vector<ngraph::AxisSet> axes = {
        {1, 2},
        {0, 2}
};

const auto explicitBroadcastParams = ::testing::Combine(
        ::testing::Values(targetShapesExplicit[0]),
        ::testing::Values(axes[0]),
        ::testing::Values(ngraph::op::BroadcastType::EXPLICIT),
        ::testing::Values(inShapesExplicit[0]),
        ::testing::ValuesIn(inputPrecision),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_ExplicitBroadcastCheck,
                        KmbBroadcastLayerTest,
                        explicitBroadcastParams,
                        KmbBroadcastLayerTest::getTestCaseName);

}  // namespace
