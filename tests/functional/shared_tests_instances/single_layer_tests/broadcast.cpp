//
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/broadcast.hpp"
#include <vector>
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class VPUXBroadcastLayerTest : public BroadcastLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

class VPUXBroadcastLayerTest_VPU3700 : public VPUXBroadcastLayerTest {};
class VPUXBroadcastLayerTest_VPU3720 : public VPUXBroadcastLayerTest {};

TEST_P(VPUXBroadcastLayerTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

TEST_P(VPUXBroadcastLayerTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

// NUMPY MODE

const std::vector<InferenceEngine::Precision> inputPrecision = {InferenceEngine::Precision::FP16,
                                                                InferenceEngine::Precision::FP32};

std::vector<std::vector<size_t>> inShapesNumpy = {{3, 1}, {1, 4, 1}};

std::vector<std::vector<size_t>> targetShapesNumpy = {{2, 3, 6}, {1, 4, 4}};

const auto numpyBroadcastParams1 = ::testing::Combine(
        ::testing::Values(targetShapesNumpy[0]), ::testing::Values(ngraph::AxisSet{}),  // not used in numpy mode
        ::testing::Values(ngraph::op::BroadcastType::NUMPY), ::testing::Values(inShapesNumpy[0]),
        ::testing::ValuesIn(inputPrecision), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_NumpyBroadcastCheck1, VPUXBroadcastLayerTest_VPU3700, numpyBroadcastParams1,
                        VPUXBroadcastLayerTest_VPU3700::getTestCaseName);

const auto numpyBroadcastParams2 = ::testing::Combine(
        ::testing::Values(targetShapesNumpy[1]), ::testing::Values(ngraph::AxisSet{}),
        ::testing::Values(ngraph::op::BroadcastType::NUMPY), ::testing::Values(inShapesNumpy[1]),
        ::testing::ValuesIn(inputPrecision), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_NumpyBroadcastCheck2, VPUXBroadcastLayerTest_VPU3700, numpyBroadcastParams2,
                        VPUXBroadcastLayerTest_VPU3700::getTestCaseName);

// BIDIRECTIONAL MODE

std::vector<std::vector<size_t>> inShapesBidi = {{4, 1}, {1, 4, 1}, {4, 1, 1}};

std::vector<std::vector<size_t>> targetShapesBidi = {{2, 1, 4}, {1, 4, 4}, {1, 1, 2, 2}};

const auto bidirectionalBroadcastParams1 = ::testing::Combine(
        ::testing::Values(targetShapesBidi[0]), ::testing::Values(ngraph::AxisSet{}),  // not used in bidirectional mode
        ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL), ::testing::Values(inShapesBidi[0]),
        ::testing::ValuesIn(inputPrecision), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck1, VPUXBroadcastLayerTest_VPU3700,
                        bidirectionalBroadcastParams1, VPUXBroadcastLayerTest_VPU3700::getTestCaseName);

const auto bidirectionalBroadcastParams2 = ::testing::Combine(
        ::testing::Values(targetShapesBidi[1]), ::testing::Values(ngraph::AxisSet{}),
        ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL), ::testing::Values(inShapesBidi[1]),
        ::testing::ValuesIn(inputPrecision), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck2, VPUXBroadcastLayerTest_VPU3700,
                        bidirectionalBroadcastParams2, VPUXBroadcastLayerTest_VPU3700::getTestCaseName);

const auto bidirectionalBroadcastParams3 = ::testing::Combine(
        ::testing::Values(targetShapesBidi[2]), ::testing::Values(ngraph::AxisSet{}),
        ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL), ::testing::Values(inShapesBidi[2]),
        ::testing::ValuesIn(inputPrecision), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck3, VPUXBroadcastLayerTest_VPU3700,
                        bidirectionalBroadcastParams3, VPUXBroadcastLayerTest_VPU3700::getTestCaseName);

// EXPLICIT MODE

std::vector<std::vector<size_t>> inShapesExplicit = {{3, 1}, {2, 4}};

std::vector<std::vector<size_t>> targetShapesExplicit = {{2, 3, 1}, {2, 3, 4}};

std::vector<ngraph::AxisSet> axes = {{1, 2}, {0, 2}};

const auto explicitBroadcastParams1 = ::testing::Combine(
        ::testing::Values(targetShapesExplicit[0]), ::testing::Values(axes[0]),
        ::testing::Values(ngraph::op::BroadcastType::EXPLICIT), ::testing::Values(inShapesExplicit[0]),
        ::testing::ValuesIn(inputPrecision), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_ExplicitBroadcastCheck1, VPUXBroadcastLayerTest_VPU3700, explicitBroadcastParams1,
                        VPUXBroadcastLayerTest_VPU3700::getTestCaseName);

const auto explicitBroadcastParams2 = ::testing::Combine(
        ::testing::Values(targetShapesExplicit[1]), ::testing::Values(axes[1]),
        ::testing::Values(ngraph::op::BroadcastType::EXPLICIT), ::testing::Values(inShapesExplicit[1]),
        ::testing::ValuesIn(inputPrecision), ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_CASE_P(smoke_ExplicitBroadcastCheck2, VPUXBroadcastLayerTest_VPU3700, explicitBroadcastParams2,
                        VPUXBroadcastLayerTest_VPU3700::getTestCaseName);

// VPU3720 instantiation

INSTANTIATE_TEST_CASE_P(smoke_precommit_NumpyBroadcastCheck1_VPU3720, VPUXBroadcastLayerTest_VPU3720,
                        numpyBroadcastParams1, VPUXBroadcastLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_ExplicitBroadcastCheck1_VPU3720, VPUXBroadcastLayerTest_VPU3720,
                        explicitBroadcastParams1, VPUXBroadcastLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_BidirectionalBroadcastCheck1_VPU3720, VPUXBroadcastLayerTest_VPU3720,
                        bidirectionalBroadcastParams1, VPUXBroadcastLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_NumpyBroadcastCheck2_VPU3720, VPUXBroadcastLayerTest_VPU3720, numpyBroadcastParams2,
                        VPUXBroadcastLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ExplicitBroadcastCheck2_VPU3720, VPUXBroadcastLayerTest_VPU3720, explicitBroadcastParams2,
                        VPUXBroadcastLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck2_VPU3720, VPUXBroadcastLayerTest_VPU3720,
                        bidirectionalBroadcastParams2, VPUXBroadcastLayerTest_VPU3720::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BidirectionalBroadcastCheck3_VPU3720, VPUXBroadcastLayerTest_VPU3720,
                        bidirectionalBroadcastParams3, VPUXBroadcastLayerTest_VPU3720::getTestCaseName);

}  // namespace
