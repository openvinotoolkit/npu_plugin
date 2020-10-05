// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/mvn.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbMvnLayerTest : public MvnLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

TEST_P(KmbMvnLayerTest, basicTest) {
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

const std::vector<std::vector<size_t>> inputShapes = {
    {1, 32, 17},
    {1, 37, 9},
    {1, 16, 5, 8},
    {2, 19, 5, 10},
    {7, 32, 2, 8},
    {5, 8, 3, 5},
    {4, 41, 6, 9},
    {1, 32, 8, 1, 6},
    {1, 9, 1, 15, 9},
    {6, 64, 6, 1, 18},
    {2, 31, 2, 9, 1},
    {10, 16, 5, 10, 6}
};

const std::vector<bool> acrossChannels = {
    true,
    false
};

const std::vector<bool> normalizeVariance = {
    true,
    false
};

const std::vector<double> epsilon = {
    0.000000001
};

const auto MvnCases = ::testing::Combine(
    ::testing::ValuesIn(inputShapes),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::ValuesIn(acrossChannels),
    ::testing::ValuesIn(normalizeVariance),
    ::testing::ValuesIn(epsilon),
    ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)
);

// Tests fail with errors:
// 1. C++ exception with description "Size of dims(3) and format(NHWC) are inconsistent.
// openvino/inference-engine/src/inference_engine/ie_layouts.cpp:138" thrown in the test body.
// 2. C++ exception with description "MVN layer is not supported by kmbPlugin
// kmb-plugin/src/frontend_mcm/src/frontend_mcm.cpp:1533
// openvino/inference-engine/include/details/ie_exception_conversion.hpp:64" thrown in the test body.
// [Track number: S#40096]
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_TestsMVN, KmbMvnLayerTest, MvnCases, KmbMvnLayerTest::getTestCaseName);
