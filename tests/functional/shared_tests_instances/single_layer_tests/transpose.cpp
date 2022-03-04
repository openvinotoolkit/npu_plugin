// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/transpose.hpp"
#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbTransposeLayerTest: public TransposeLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {};
class KmbTransposeLayerTest_MLIR : public KmbTransposeLayerTest {};
class KmbTransposeLayerTest_MTL : public KmbTransposeLayerTest {
    void SkipBeforeLoad() override {
        if (std::getenv("OV_BUILD_DIR") == nullptr) {
            throw LayerTestsUtils::KmbSkipTestException(
                    "OV_BUILD_DIR env directory must be specified, in order to reach act-shave kernels.");
        }

#if defined(__arm__) || defined(__aarch64__) || defined(_WIN32) || defined(_WIN64)
        throw LayerTestsUtils::KmbSkipTestException("Does not compile on ARM and Windows.");
#endif
    }

    void SkipBeforeInfer() override {
#ifndef ENABLE_IMD_BACKEND
        throw LayerTestsUtils::KmbSkipTestException("Runtime issue.");
#endif
    }
};

TEST_P(KmbTransposeLayerTest, CompareWithRefs) {
    Run();
}

TEST_P(KmbTransposeLayerTest_MLIR, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbTransposeLayerTest_MTL, MLIR_MTL) {
    useCompilerMLIR();
    setPlatformMTL();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions


using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,
};

// MCM instantiation

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

INSTANTIATE_TEST_SUITE_P(
    smoke_Transpose,
    KmbTransposeLayerTest,
    params,
    KmbTransposeLayerTest::getTestCaseName);

// MLIR 2D instantiation

const std::vector<std::vector<size_t>> inputShapes2D = {
    std::vector<size_t>{50, 100},
};

const std::vector<std::vector<size_t>> inputOrder2D = {
    std::vector<size_t>{},
};

const auto params2D = testing::Combine(
    testing::ValuesIn(inputOrder2D),
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inputShapes2D),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

// [Track number: W#7312]
INSTANTIATE_TEST_SUITE_P(
    DISABLED_smoke_Transpose2D,
    KmbTransposeLayerTest_MLIR,
    params2D,
    KmbTransposeLayerTest::getTestCaseName);

// MLIR 4D instantiation

const std::vector<std::vector<size_t>> inputShapes4D = {
    std::vector<size_t>{1, 3, 100, 100},
};

const std::vector<std::vector<size_t>> inputOrder4D = {
    std::vector<size_t>{0, 3, 2, 1},

    // [Track number: W#7311]
    // std::vector<size_t>{},
};

const auto params4D = testing::Combine(
    testing::ValuesIn(inputOrder4D),
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inputShapes4D),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_SUITE_P(
    smoke_Transpose4D,
    KmbTransposeLayerTest_MLIR,
    params4D,
    KmbTransposeLayerTest::getTestCaseName);

// MLIR 4D MemPermute instantiation

const std::vector<std::vector<size_t>> inputShapesMemPerm = {
        std::vector<size_t>{1, 3, 100, 100},
};

const std::vector<std::vector<size_t>> inputOrderMemPerm = {
        std::vector<size_t>{0, 2, 3, 1},
};

const auto paramsMemPermNCHWtoNHWC = testing::Combine(
        testing::ValuesIn(inputOrderMemPerm),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Layout::NCHW),
        testing::Values(InferenceEngine::Layout::NHWC),
        testing::ValuesIn(inputShapesMemPerm),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(
        smoke_TransposeMemPermNCHW,
        KmbTransposeLayerTest_MLIR,
        paramsMemPermNCHWtoNHWC,
        KmbTransposeLayerTest::getTestCaseName);

const auto paramsMemPermInNHWC = testing::Combine(
        testing::ValuesIn(inputOrderMemPerm),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::NHWC),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputShapesMemPerm),
        testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(
        smoke_TransposeMemPermNHWC,
        KmbTransposeLayerTest_MLIR,
        paramsMemPermInNHWC,
        KmbTransposeLayerTest::getTestCaseName);

// ------ MTL ------

const std::vector<std::vector<size_t>> inputOrderMTL = {
        std::vector<size_t>{0, 2, 3, 1},
};

const auto paramsMTL = testing::Combine(
    testing::ValuesIn(inputOrderMTL),
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::FP16),
    testing::Values(InferenceEngine::Precision::FP16),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inputShapesMemPerm),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(
        smoke_TransposeMTL,
        KmbTransposeLayerTest_MTL,
        paramsMTL,
        KmbTransposeLayerTest::getTestCaseName);

}  // namespace
