// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/log_softmax.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class VPUXLogSoftmaxLayerTest: public LogSoftmaxLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

    TEST_P(VPUXLogSoftmaxLayerTest, CompareWithRefs) {
        threshold = 1e-3;
        useCompilerMLIR();
        Run();
    }
}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP16
    };

    const std::vector<InferenceEngine::Layout> inLayouts2D = {
            InferenceEngine::Layout::NC,
    };
    const std::vector<InferenceEngine::Layout> outLayouts2D = {
            InferenceEngine::Layout::NC
    };

    std::vector<std::vector<size_t>> inShapes2D = {
            {1200, 5},
    };

    std::vector<int64_t> axis2D = {1};

    const std::vector<InferenceEngine::Precision> inputPrecisions = {
            InferenceEngine::Precision::FP16
    };

    const std::vector<InferenceEngine::Precision> outputPrecisions = {
            InferenceEngine::Precision::FP16
    };

    const auto params2D = testing::Combine(
            testing::ValuesIn(netPrecisions), testing::ValuesIn(inputPrecisions), testing::ValuesIn(outputPrecisions), testing::ValuesIn(inLayouts2D), testing::ValuesIn(outLayouts2D),
            testing::ValuesIn(inShapes2D), testing::ValuesIn(axis2D),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice),::testing::Values(std::map<std::string, std::string>({})));

    INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax, VPUXLogSoftmaxLayerTest, params2D, VPUXLogSoftmaxLayerTest::getTestCaseName);

}  // namespace
