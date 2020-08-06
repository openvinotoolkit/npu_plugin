// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/convolution.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbConvolutionLayerTest: public ConvolutionLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbConvolutionLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16};

/* ============= 2D Convolution ============= */
/* original values were:
const std::vector<InferenceEngine::SizeVector> dilations = {{1, 1}, {3, 1}};
*/
const std::vector<InferenceEngine::SizeVector> kernels = {{3, 3}, {3, 5}};
const std::vector<InferenceEngine::SizeVector> strides = {{1, 1}, {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}, {0, 3}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}, {0, 3}};
const std::vector<InferenceEngine::SizeVector> dilations = {{1, 1}};
const std::vector<size_t> numOutCannels = {1, 5};
const std::vector<ngraph::op::PadType> padTypes = {ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID};

const auto conv2DParams_ExplicitPadding = ::testing::Combine(::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds), ::testing::ValuesIn(dilations),
    ::testing::ValuesIn(numOutCannels), ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})), ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutCannels), ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(Convolution2D_ExplicitPadding, KmbConvolutionLayerTest,
    ::testing::Combine(conv2DParams_ExplicitPadding,
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::SizeVector({1, 3, 30, 30})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Convolution2D_AutoPadValid, KmbConvolutionLayerTest,
    ::testing::Combine(conv2DParams_AutoPadValid,
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::SizeVector({1, 3, 30, 30})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
    ConvolutionLayerTest::getTestCaseName);

/* ============= 3D Convolution ============= */
const std::vector<InferenceEngine::SizeVector> kernels3d = {{3, 3, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0}};

const std::vector<InferenceEngine::SizeVector> strides3d = {{1, 1, 1}};
const std::vector<InferenceEngine::SizeVector> dilations3d = {{1, 1, 1}};
const std::vector<size_t> numOutChannels3d = {5};

/* original values were:
const std::vector<InferenceEngine::SizeVector> kernels3d = {{3, 3, 3}, {3, 5, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0}, {0, 2, 0}};

const std::vector<InferenceEngine::SizeVector> strides3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<InferenceEngine::SizeVector> dilations3d = {{1, 1, 1}, {1, 2, 1}};
*/

const auto conv3DParams_ExplicitPadding = ::testing::Combine(::testing::ValuesIn(kernels3d),
    ::testing::ValuesIn(strides3d), ::testing::ValuesIn(paddings3d), ::testing::ValuesIn(paddings3d),
    ::testing::ValuesIn(dilations3d), ::testing::ValuesIn(numOutChannels3d), ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto conv3DParams_AutoPadValid =
    ::testing::Combine(::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::ValuesIn(dilations3d), ::testing::ValuesIn(numOutChannels3d), ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(DISABLED_Convolution3D_ExplicitPadding, KmbConvolutionLayerTest,
    ::testing::Combine(conv3DParams_ExplicitPadding,
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::SizeVector({1, 3, 10, 10, 10})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(DISABLED_Convolution3D_AutoPadValid, KmbConvolutionLayerTest,
    ::testing::Combine(conv3DParams_AutoPadValid,
        ::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::SizeVector({1, 3, 10, 10, 10})),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
    ConvolutionLayerTest::getTestCaseName);
}  // namespace
