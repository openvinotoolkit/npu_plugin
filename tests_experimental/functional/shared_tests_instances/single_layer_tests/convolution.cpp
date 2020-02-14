// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/convolution.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

// Common params
const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
//         InferenceEngine::Precision::FP16, // "[NOT_IMPLEMENTED] Input image format FP16 is not supported yet...
        InferenceEngine::Precision::U8,
//         InferenceEngine::Precision::I8 // Too much cases
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

/* ============= 2D Convolution ============= */
const std::vector<InferenceEngine::SizeVector> kernels = {{3, 3},
                                                          {3, 5}};
const std::vector<InferenceEngine::SizeVector> strides = {{1, 1},
                                                          {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0},
                                                       {0, 3}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0},
                                                     {0, 3}};
const std::vector<InferenceEngine::SizeVector> dilations = {{1, 1},
                                                            {3, 1}};
const std::vector<size_t> numOutCannels = {1, 5};
const std::vector<ngraph::op::PadType> padTypes = {
        ngraph::op::PadType::EXPLICIT,
        ngraph::op::PadType::VALID
};

const auto conv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutCannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto conv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutCannels),
        ::testing::Values(ngraph::op::PadType::VALID)
);

INSTANTIATE_TEST_CASE_P(Convolution2D_ExplicitPadding, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv2DParams_ExplicitPadding,
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::SizeVector({1, 3, 30, 30})),
                                ::testing::Values(InferenceEngine::SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                        ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Convolution2D_AutoPadValid, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv2DParams_AutoPadValid,
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::SizeVector({1, 3, 30, 30})),
                                ::testing::Values(InferenceEngine::SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 3D Convolution ============= */
const std::vector<InferenceEngine::SizeVector> kernels3d = {{3, 3, 3},
                                                            {3, 5, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0},
                                                        {0, 2, 0}};

const std::vector<InferenceEngine::SizeVector> strides3d = {{1, 1, 1},
                                                            {1, 2, 1}};
const std::vector<InferenceEngine::SizeVector> dilations3d = {{1, 1, 1},
                                                              {1, 2, 1}};

const auto conv3DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::Values(5),
        ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto conv3DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::ValuesIn(dilations3d),
        ::testing::Values(5),
        ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(Convolution3D_ExplicitPadding, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv3DParams_ExplicitPadding,
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::SizeVector({1, 3, 10, 10, 10})),
                                ::testing::Values(InferenceEngine::SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                        ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Convolution3D_AutoPadValid, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv3DParams_AutoPadValid,
                                ::testing::ValuesIn(inputPrecisions),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::SizeVector({1, 3, 10, 10, 10})),
                                ::testing::Values(InferenceEngine::SizeVector()),
                                ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                        ConvolutionLayerTest::getTestCaseName);

const std::vector<InferenceEngine::SizeVector> targetReshapeShapes = {{2, 3, 10, 10, 10},
                                                                      {1, 3, 12, 12, 12},
                                                                      {2, 3, 12, 13, 14}};

/* ============= 2D Convolution Reshape ============= */
const auto convReshape = ::testing::Combine(
        ::testing::Values(InferenceEngine::SizeVector({3, 3, 3})),
        ::testing::Values(InferenceEngine::SizeVector({1, 1, 1})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(InferenceEngine::SizeVector({1, 1, 1})),
        ::testing::Values(5),
        ::testing::ValuesIn(padTypes));

const auto convReshapefullParamsProduct = ::testing::Combine(
        convReshape,
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::SizeVector({1, 3, 10, 10, 10})),
        ::testing::ValuesIn(targetReshapeShapes),
        ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY));

INSTANTIATE_TEST_CASE_P(Reshape, ConvolutionLayerTest, convReshapefullParamsProduct,
                        ConvolutionLayerTest::getTestCaseName);

const auto convReshapeNegativefullParamsProduct = ::testing::Combine(
        convReshape,
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(InferenceEngine::SizeVector({1, 3, 10, 10, 10})),
        ::testing::Values(InferenceEngine::SizeVector({1, 10, 10, 10, 10})),
        ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY));

INSTANTIATE_TEST_CASE_P(ReshapeNegative, NegativeReshapeConvolutionSingleLayerTest,
                        convReshapeNegativefullParamsProduct,
                        ConvolutionLayerTest::getTestCaseName);

}  // namespace
