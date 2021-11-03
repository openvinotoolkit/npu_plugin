// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/cum_sum.hpp"
#include "common_test_utils/test_constants.hpp"

#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbCumSumLayerTest : public CumSumLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbCumSumLayerTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

const std::vector<std::vector<size_t>> shapes = {
    {16},
    {9, 15},
    {16, 10, 12},
    {5, 14, 5, 7},
    {7, 8, 6, 7, 13},
};

const std::vector<InferenceEngine::Precision> inputPrecision = {
    InferenceEngine::Precision::FP16
};

const std::vector<int64_t> axes = {0, 1, 2, 3, 4};
const std::vector<int64_t> negativeAxes = {-1, -2, -3, -4};

const std::vector<bool> exclusive = {true, false};
const std::vector<bool> reverse = {true, false};

const auto testCasesNegativeAxis = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{4, 16, 3, 6, 5}),
    ::testing::ValuesIn(inputPrecision),
    ::testing::ValuesIn(negativeAxes),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

const auto testCasesAxis_0 = ::testing::Combine(
    ::testing::ValuesIn(shapes),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(axes[0]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

const auto testCasesAxis_1 = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(shapes.begin() + 1, shapes.end())),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(axes[1]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

const auto testCasesAxis_2 = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(shapes.begin() + 2, shapes.end())),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(axes[2]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

const auto testCasesAxis_3 = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(shapes.begin() + 3, shapes.end())),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(axes[3]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

const auto testCasesAxis_4 = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(shapes.begin() + 4, shapes.end())),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(axes[4]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_SUITE_P(smoke_CumSumTests_negative_axis, KmbCumSumLayerTest, testCasesNegativeAxis, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSumTests_axis_0, KmbCumSumLayerTest, testCasesAxis_0, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSumTests_axis_1, KmbCumSumLayerTest, testCasesAxis_1, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSumTests_axis_2, KmbCumSumLayerTest, testCasesAxis_2, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSumTests_axis_3, KmbCumSumLayerTest, testCasesAxis_3, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CumSumTests_axis_4, KmbCumSumLayerTest, testCasesAxis_4, CumSumLayerTest::getTestCaseName);
