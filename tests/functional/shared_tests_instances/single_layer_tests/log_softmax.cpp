// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/log_softmax.hpp"
#include <vector>
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbLogSoftmaxLayerTest : public LogSoftmaxLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

// Disabled as 'convert-subtract-to-negative-add' pass is not ready for one/more platforms in `ReferenceSW` mode
// These tests shall be re-enabled and revalidate once such pass is added to 'ReferenceSW' pipeline
TEST_P(KmbLogSoftmaxLayerTest, DISABLED_CompareWithRefs_SW_MLIR) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}
TEST_P(KmbLogSoftmaxLayerTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecision = {InferenceEngine::Precision::FP16};

const std::vector<InferenceEngine::Precision> dataPrecision = {InferenceEngine::Precision::UNSPECIFIED};

/* ============= 2D LogSoftmax ============= */

const std::vector<InferenceEngine::Layout> layouts2D = {InferenceEngine::Layout::NC};

std::vector<std::vector<size_t>> inShapes2D = {
        {12, 5}, {1200, 5}  // real case
};

std::vector<int64_t> axis2D = {1};

const auto params2D = testing::Combine(
        testing::ValuesIn(netPrecision), testing::ValuesIn(dataPrecision), testing::ValuesIn(dataPrecision),
        testing::ValuesIn(layouts2D), testing::ValuesIn(layouts2D), testing::ValuesIn(inShapes2D),
        testing::ValuesIn(axis2D), testing::Values(LayerTestsUtils::testPlatformTargetDevice),
        ::testing::Values(std::map<std::string, std::string>({})));

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_2D, KmbLogSoftmaxLayerTest, params2D,
                         KmbLogSoftmaxLayerTest::getTestCaseName);

/* ============= 3D/4D LogSoftmax ============= */

const std::vector<InferenceEngine::Layout> layouts = {InferenceEngine::Layout::ANY};
//
// [Track number: E#37278]
//
std::vector<std::vector<size_t>> inShapes = {
        {1, 20, 256, 512}, {1, 10, 256, 512},
        // {5, 30, 1}
};

std::vector<int64_t> axis = {2, 3};
//
// [Track number: E#37278]
//
//     std::vector<int64_t> axis = {0, 1, 2, 3};
//     std::vector<int64_t> axis = {0, 1, 2};

const auto params = testing::Combine(testing::ValuesIn(netPrecision), testing::ValuesIn(dataPrecision),
                                     testing::ValuesIn(dataPrecision), testing::ValuesIn(layouts),
                                     testing::ValuesIn(layouts), testing::ValuesIn(inShapes), testing::ValuesIn(axis),
                                     testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                     ::testing::Values(std::map<std::string, std::string>({})));

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax_3D_4D, KmbLogSoftmaxLayerTest, params,
                         KmbLogSoftmaxLayerTest::getTestCaseName);
}  // namespace
