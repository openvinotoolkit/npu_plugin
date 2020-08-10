// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/quantized_group_convolution.hpp"
#include "common_test_utils/test_constants.hpp"

#include <vector>

#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbQuantGroupConvLayerTest: public QuantGroupConvLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbQuantGroupConvLayerTest, CompareWithRefs) {
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16
};


const std::vector<size_t> numOutChannels = {3, 24, 48};
const std::vector<size_t> numGroups = {3};

const std::vector<size_t > levels = {256};
const std::vector<QuantizationGranularity> granularity = {Pertensor, Perchannel};
const std::vector<bool> quantizeWeights2D = {false};

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<size_t >> inputShapes2D = {{1, 3, 10, 10}, {1, 24, 10, 10}};
const std::vector<std::vector<size_t >> kernels2D = {{1, 1}, {3, 3}};
const std::vector<std::vector<size_t >> strides2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}};
const std::vector<std::vector<size_t >> dilations2D = {{1, 1}};


const auto quantGroupConv2DParams = ::testing::Combine(
        ::testing::ValuesIn(kernels2D),
        ::testing::ValuesIn(strides2D),
        ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D),
        ::testing::ValuesIn(dilations2D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(granularity),
        ::testing::ValuesIn(quantizeWeights2D)
);

INSTANTIATE_TEST_CASE_P(QuantGroupConv2D, KmbQuantGroupConvLayerTest,
                        ::testing::Combine(
                                quantGroupConv2DParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(inputShapes2D),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        QuantGroupConvLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<std::vector<size_t >> inputShapes3D = {{1, 3, 5, 5, 5}, {1, 24, 5, 5, 5}};
const std::vector<std::vector<size_t >> kernels3D = {{3, 3, 3}};
const std::vector<std::vector<size_t >> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{0, 0, 0}};
const std::vector<std::vector<size_t >> dilations3D = {{1, 1, 1}};
const std::vector<bool> quantizeWeights3D = {true};

const auto quantGroupConv3DParams = ::testing::Combine(
        ::testing::ValuesIn(kernels3D),
        ::testing::ValuesIn(strides3D),
        ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D),
        ::testing::ValuesIn(dilations3D),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(granularity),
        ::testing::ValuesIn(quantizeWeights3D)
);

INSTANTIATE_TEST_CASE_P(DISABLED_QuantGroupConv3D, KmbQuantGroupConvLayerTest,
                        ::testing::Combine(
                                quantGroupConv3DParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(inputShapes3D),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        QuantGroupConvLayerTest::getTestCaseName);

}  // namespace
