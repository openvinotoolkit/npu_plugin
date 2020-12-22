// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/transpose.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

    class KmbTransposeLayerTest: public TransposeLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};

    TEST_P(KmbTransposeLayerTest, CompareWithRefs) {
        Run();
    }
}  // namespace LayerTestsDefinitions


using namespace LayerTestsDefinitions;

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
    };

    const std::vector<std::vector<size_t>> inputShapes = {
            std::vector<size_t>{1, 3, 100, 100},
    };

    const std::vector<std::vector<size_t>> inputOrder = {
            std::vector<size_t>{0, 3, 2, 1},
            // std::vector<size_t>{}, - It is from CPU-plugin code.
            // This empty vector leads to SIGFPE (Arithmetic exception)
            // at kmb-plugin/src/mcmCompiler/src/tensor/tensor.cpp:299
            // inside method void mv::Tensor::populate(const std::vector<int64_t>& data)
            // there is a blockSize_ == 0 and expression auto dataSize = shape_.totalSize()/blockSize_;
            // so we get division on zero.
            // [Track number: S#45017]
    };

    const auto params = testing::Combine(
            testing::ValuesIn(inputOrder),
            testing::ValuesIn(netPrecisions),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::Values(InferenceEngine::Layout::ANY),
            testing::ValuesIn(inputShapes),
            testing::Values(LayerTestsUtils::testPlatformTargetDevice)
    );

    INSTANTIATE_TEST_CASE_P(
            smoke_Transpose,
            KmbTransposeLayerTest,
            params,
            KmbTransposeLayerTest::getTestCaseName
    );

}  // namespace
