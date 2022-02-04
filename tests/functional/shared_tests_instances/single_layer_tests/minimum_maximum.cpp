// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/minimum_maximum.hpp"
#include "kmb_layer_test.hpp"
#include <common/functions.h>

namespace LayerTestsDefinitions {

class KmbMaxMinLayerTest: public MaxMinLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon { };

class KmbMaxMinLayerTest_MCM: public KmbMaxMinLayerTest {
    void SkipBeforeInfer() override {
        // [Track number: E#20948]
        const auto testName =
            std::string{::testing::UnitTest::GetInstance()->current_test_info()->test_case_name()};
        const auto isSmokeMaxScalar = testName.find("smoke_maximum_scalar") != std::string::npos;
        const auto isLevel0 = getBackendName(*getCore()) == "LEVEL0";
        if (isSmokeMaxScalar && isLevel0 && isCompilerMCM()) {
            throw LayerTestsUtils::KmbSkipTestException("Level0: sporadic failure on device");
        }
    }
 };

class KmbMaxMinLayerTest_MLIR: public KmbMaxMinLayerTest { };

class KmbMaxMinLayerTestMLIR_MTL: public KmbMaxMinLayerTest {
    void SetUp() override {
        KmbMaxMinLayerTest::SetUp();

        inPrc = InferenceEngine::Precision::FP16;
        outPrc = InferenceEngine::Precision::FP16;
    }
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
        throw LayerTestsUtils::KmbSkipTestException("Runtime issue.");
    }
 };

// MCM and MLIR tests use different parameters. See below.

TEST_P(KmbMaxMinLayerTest_MCM, CompareWithRefs) {
    Run();
}

TEST_P(KmbMaxMinLayerTest_MLIR, CompareWithRefs) {
    useCompilerMLIR();
    Run();
}

TEST_P(KmbMaxMinLayerTestMLIR_MTL, CompareWithRefs_MLIR_MTL) {
    useCompilerMLIR();
    setPlatformMTL();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {
const std::vector<std::vector<std::vector<size_t>>> inShapes4D = {
        {{1,64,32,32}, {1,64,32,32}},
        {{1, 1, 1, 3}, {1}}
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16,
};

const std::vector<ngraph::helpers::MinMaxOpType> opType = {
        ngraph::helpers::MinMaxOpType::MINIMUM,
        ngraph::helpers::MinMaxOpType::MAXIMUM,
};

const std::vector<ngraph::helpers::InputLayerType> inputType = {
        ngraph::helpers::InputLayerType::CONSTANT
};

const std::vector<InferenceEngine::Layout> layout4D = {
        InferenceEngine::Layout::NCHW,
        // NHCW layout kernel is not being tested
        // Eltwise NHWC layers are failing to infer
        // [Track number: E#25740]
        InferenceEngine::Layout::NHWC
};

INSTANTIATE_TEST_SUITE_P(smoke_maximum_4D, KmbMaxMinLayerTest_MLIR,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes4D),
                                ::testing::ValuesIn(opType),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::ValuesIn(layout4D),
                                ::testing::ValuesIn(layout4D),
                                ::testing::ValuesIn(inputType),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbMaxMinLayerTest::getTestCaseName);

const std::vector<std::vector<std::vector<size_t>>> inShapes3D = {
        {{1, 2, 4}, {1}}
};

INSTANTIATE_TEST_SUITE_P(smoke_maximum_3D, KmbMaxMinLayerTest_MLIR,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes3D),
                                ::testing::ValuesIn(opType),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputType),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbMaxMinLayerTest::getTestCaseName);

const std::vector<std::vector<std::vector<size_t>>> inShapesScalar = {
        /// test scalar constant input for case MAX(x, scalar_threshold)
        {{32}, {1}}
};

// [Track number: E#13808]
// [Track number: S#43484]
INSTANTIATE_TEST_SUITE_P(smoke_maximum_scalar, KmbMaxMinLayerTest_MCM,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapesScalar),
                                ::testing::ValuesIn(opType),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputType),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbMaxMinLayerTest::getTestCaseName);

const std::vector<std::vector<std::vector<size_t>>> inShapes3D_MTL = {
        {{2, 2, 2}, {2, 2, 2}}
};

INSTANTIATE_TEST_SUITE_P(smoke_maximum_3D_MTL, KmbMaxMinLayerTestMLIR_MTL,
                        ::testing::Combine(
                                ::testing::ValuesIn(inShapes3D_MTL),
                                ::testing::ValuesIn(opType),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::FP16),
                                ::testing::Values(InferenceEngine::Precision::FP16),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputType),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        KmbMaxMinLayerTest::getTestCaseName);

}  // namespace
