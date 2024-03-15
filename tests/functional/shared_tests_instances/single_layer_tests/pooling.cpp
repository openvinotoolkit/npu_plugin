// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_layer_tests/pooling.hpp"
#include "vpux_private_properties.hpp"

#include <vector>

#include <common/functions.h>
#include "vpu_ov1_layer_test.hpp"

namespace LayerTestsDefinitions {

// Option added base on CI request to decrease test runtime
// Important to enable macro (remove //) to run full tests in CI every time your change can impact AVG/MAX pool.
// Both operations are transformed in some scenario to NCE task, so it is important to enable testing when touch any of
// this mlir passes.
// #define ENABLE_ALL_POOL_TESTS

#ifdef ENABLE_ALL_POOL_TESTS
#define INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(A, B, C, D) INSTANTIATE_TEST_SUITE_P(A, B, C, D)
#else
#define INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(A, B, C, D) INSTANTIATE_TEST_SUITE_P(DISABLED_##A, B, C, D)
#endif

class PoolingLayerTest_NPU3700 : public PoolingLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SkipBeforeLoad() override {
        const auto& poolParams = std::get<0>(GetParam());

        ngraph::helpers::PoolingTypes poolType;
        std::vector<size_t> strides;
        ngraph::op::RoundingType roundingMode;
        std::tie(poolType, std::ignore, strides, std::ignore, std::ignore, roundingMode, std::ignore, std::ignore) =
                poolParams;

        if (poolType == ngraph::helpers::PoolingTypes::AVG &&
            configuration[ov::intel_vpux::compilation_mode.name()] == "DefaultHW") {
            threshold = 0.25;
        }

        // MLIR uses software layer, which seem to be flawed
        if (poolType == ngraph::helpers::PoolingTypes::AVG) {
            if (strides[0] != 1 || strides[1] != 1) {
                throw LayerTestsUtils::VpuSkipTestException("AVG pool strides != 1 produces inaccurate results");
            }
        }
    }
};

TEST_P(PoolingLayerTest_NPU3700, SW) {
    setPlatformVPU3700();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(PoolingLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

class PoolingLayerTest_NPU3720 : public PoolingLayerTest, virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SkipBeforeLoad() override {
        std::vector<size_t> inputShapes;
        std::tie(std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore, inputShapes,
                 std::ignore) = GetParam();
        const auto& poolParams = std::get<0>(GetParam());
        ngraph::helpers::PoolingTypes poolType;
        std::vector<size_t> kernel;
        std::vector<size_t> strides;
        std::vector<size_t> padBegin;
        std::vector<size_t> padEnd;
        ngraph::op::RoundingType roundingMode;
        ngraph::op::PadType padType;
        bool excludePad;
        std::tie(poolType, kernel, strides, padBegin, padEnd, roundingMode, padType, excludePad) = poolParams;

        // all DefaultHW test Should be enable when E#94485 will be fixed. Convert to HW scenario not implemented
        if ((poolType == ngraph::helpers::PoolingTypes::AVG) &&
            (configuration[ov::intel_vpux::compilation_mode.name()] == "DefaultHW") && (strides.size() == 2)) {
            // support exclude pad for reduce number of scenario, when HandleExcludePadForAvgPoolPass fail,
            // excludePad remain, should not validate ConvertIEToVPUNCEPass for AvgPool
            if (excludePad) {
                std::vector<size_t> ones{1, 1};
                if ((padBegin != ones) || (padEnd != ones) || (strides != ones)) {
                    throw LayerTestsUtils::VpuSkipTestException(
                            "AVGPool convert to NCE with excludePad, invalid conversion");
                }
            }
            // special implementation in reference for CEIL rounding mode with padBegin=0, padEnd=0, and Ceil rounding
            // involve padding with 1. see openvino reference:
            // openvino/src/core/reference/include/ngraph/runtime/reference/avg_pool.hpp
            // if all are 0, and CEIL request in fact padding, then enable excludePad, else, if just 1 of
            // value are not 0, work as expected, divide by constant kernel size.
            // Hw should implement in same way, or allow go to SW implementation.
            if (roundingMode == ngraph::op::RoundingType::CEIL) {
                std::vector<size_t> zeros{0, 0};
                if ((padBegin == zeros) && (padEnd == zeros)) {
                    throw LayerTestsUtils::VpuSkipTestException(
                            "AVG pool CEIL rounding with PADS 0 are not proper converted to NCE AvgPool");
                }
            }
            // Default HW pipe produce wrong values for this combination, if input size is just 8x8 that produce 3x3 or
            // 1x1 output, SW reference pipeline pass. Probably align issue in HW version. Padding mode is valid
            if ((inputShapes[3] <= 8) && (inputShapes[2] <= 8) && (strides[0] == 2) && (strides[1] == 2) &&
                (padType == ngraph::op::PadType::VALID)) {
                throw LayerTestsUtils::VpuSkipTestException(
                        "AVG Pool VALID pad type for small resolution invalid conversion to NCE AvgPool");
            }
        }

        // Invalid padding with 0 for MaxPool, should be -MaxFloat
        // src/vpux_compiler/src/dialect/IE/passes/handle_large_pads.cpp pad with 0 as for Avg, but Max is not the same.
        // Remove when  E#99182 will be fixed. Or open a separate ticket related to E#69906
        if ((poolType == ngraph::helpers::PoolingTypes::MAX) &&
            (configuration[ov::intel_vpux::compilation_mode.name()] == "DefaultHW") && (strides.size() == 2)) {
            size_t kernel0 = kernel[0];
            size_t kernel1 = kernel[1];
            if (roundingMode == ngraph::op::RoundingType::CEIL) {
                kernel0 -= 1;
                kernel1 -= 1;
            }
            if ((padBegin[0] >= kernel0) || (padBegin[1] >= kernel1) || (padEnd[0] >= kernel0) ||
                (padEnd[1] >= kernel1)) {
                throw LayerTestsUtils::VpuSkipTestException(
                        "MAX pool, Hw NCE version produce invalid 0 values on pad area");
            }
        }
    }
};

TEST_P(PoolingLayerTest_NPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(PoolingLayerTest_NPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

using PoolingLayerTest_NPU3720_SingleCluster = PoolingLayerTest_NPU3720;

TEST_P(PoolingLayerTest_NPU3720_SingleCluster, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    setSingleClusterMode();
    useELFCompilerBackend();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace InferenceEngine;
using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

/* ============= AutoPadValid ============= */

const auto pool_AutoPadValid =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),  //
                                              ::testing::ValuesIn<SizeVector>({{3, 3}, {5, 5}}),        // kernels
                                              ::testing::ValuesIn<SizeVector>({{1, 1}, {2, 2}}),        // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),                // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),                // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),       //
                                              ::testing::Values(ngraph::op::PadType::VALID),            //
                                              ::testing::Values(false)),  // excludePad,                          //
                           ::testing::Values(Precision::FP16),            // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),     // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),     // outPrc
                           ::testing::Values(Layout::ANY),                // inLayout
                           ::testing::Values(Layout::ANY),                // outLayout
                           ::testing::ValuesIn<SizeVector>(
                                   {{1, 8, 32, 32}, {1, 16, 24, 24}, {1, 24, 16, 16}, {1, 32, 8, 8}}),  // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AutoPadValid, PoolingLayerTest_NPU3700, pool_AutoPadValid,
                         PoolingLayerTest::getTestCaseName);

/* ============= ExplicitPadding ============= */

const auto pool_ExplicitPadding =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),    //
                                              ::testing::ValuesIn<SizeVector>({{3, 3}}),                  // kernels
                                              ::testing::ValuesIn<SizeVector>({{2, 2}}),                  // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}, {1, 1}, {0, 1}}),  // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}, {1, 1}, {0, 1}}),  // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR,
                                                                ngraph::op::RoundingType::CEIL),  //
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT),   //
                                              ::testing::Values(false)),                          //
                           ::testing::Values(Precision::FP16),                                    // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),                             // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),                             // outPrc
                           ::testing::Values(Layout::ANY),                                        // inLayout
                           ::testing::Values(Layout::ANY),                                        // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 16, 30, 30}}),                    // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_ExplicitPadding, PoolingLayerTest_NPU3700, pool_ExplicitPadding,
                         PoolingLayerTest::getTestCaseName);

/* ============= AsymmetricKernel ============= */

const auto pool_AsymmetricKernel =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),  //
                                              ::testing::ValuesIn<SizeVector>({{3, 1}, {1, 3}}),        // kernels
                                              ::testing::ValuesIn<SizeVector>({{1, 1}, {2, 2}}),        // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),                // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),                // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),       //
                                              ::testing::Values(ngraph::op::PadType::VALID),            //
                                              ::testing::Values(false)),                                // excludePad
                           ::testing::Values(Precision::FP16),                                          // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),                                   // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),                                   // outPrc
                           ::testing::Values(Layout::ANY),                                              // inLayout
                           ::testing::Values(Layout::ANY),                                              // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 16, 30, 30}}),                          // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AsymmetricKernel, PoolingLayerTest_NPU3700, pool_AsymmetricKernel,
                         PoolingLayerTest::getTestCaseName);

/* ============= AsymmetricStrides ============= */

const auto pool_AsymmetricStrides =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),  //
                                              ::testing::ValuesIn<SizeVector>({{3, 3}}),                // kernels
                                              ::testing::ValuesIn<SizeVector>({{1, 2}, {2, 1}}),        // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),                // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),                // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),       //
                                              ::testing::Values(ngraph::op::PadType::VALID),            //
                                              ::testing::Values(false)),                                // excludePad
                           ::testing::Values(Precision::FP16),                                          // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),                                   // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),                                   // outPrc
                           ::testing::Values(Layout::ANY),                                              // inLayout
                           ::testing::Values(Layout::ANY),                                              // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 16, 30, 30}}),                          // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AsymmetricStrides, PoolingLayerTest_NPU3700, pool_AsymmetricStrides,
                         PoolingLayerTest::getTestCaseName);

/* ============= LargeSize ============= */

const auto pool_LargeSize1 =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                              ::testing::ValuesIn<SizeVector>({{3, 3}}),           // kernels
                                              ::testing::ValuesIn<SizeVector>({{2, 2}}),           // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                              ::testing::Values(ngraph::op::PadType::VALID),       //
                                              ::testing::Values(false)),                           // excludePad, //
                           ::testing::Values(Precision::FP16),                                     // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // outPrc
                           ::testing::Values(Layout::ANY),                                         // inLayout
                           ::testing::Values(Layout::ANY),                                         // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 64, 128, 128}}),                   // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargeSize1, PoolingLayerTest_NPU3700, pool_LargeSize1,
                         PoolingLayerTest::getTestCaseName);

const auto pool_LargeSize2 =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                              ::testing::ValuesIn<SizeVector>({{3, 3}}),           // kernels
                                              ::testing::ValuesIn<SizeVector>({{2, 2}}),           // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                              ::testing::Values(ngraph::op::PadType::VALID),       //
                                              ::testing::Values(false)),                           // excludePad
                           ::testing::Values(Precision::FP16),                                     // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // outPrc
                           ::testing::Values(Layout::ANY),                                         // inLayout
                           ::testing::Values(Layout::ANY),                                         // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 16, 256, 256}}),                   // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargeSize2, PoolingLayerTest_NPU3700, pool_LargeSize2,
                         PoolingLayerTest::getTestCaseName);

/* ============= LargeStrides ============= */

const auto pool_LargeStrides =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX),                 //
                                              ::testing::ValuesIn<SizeVector>({{3, 3}, {11, 11}}),  // kernels
                                              ::testing::ValuesIn<SizeVector>({{9, 9}}),            // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),            // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),            // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),   //
                                              ::testing::Values(ngraph::op::PadType::VALID),        //
                                              ::testing::Values(false)),                            // excludePad

                           ::testing::Values(Precision::FP16),                  // netPrc
                           ::testing::Values(Precision::FP16),                  // inPrc
                           ::testing::Values(Precision::FP16),                  // outPrc
                           ::testing::Values(Layout::ANY),                      // inLayout
                           ::testing::Values(Layout::ANY),                      // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargeStrides, PoolingLayerTest_NPU3700, pool_LargeStrides,
                         PoolingLayerTest::getTestCaseName);

/* ============= BatchN to batch1 ============= */

const auto pool_batchN = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                            ::testing::ValuesIn<SizeVector>({{1, 1}}),           // kernels
                                            ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                            ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                            ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                            ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                            ::testing::Values(ngraph::op::PadType::VALID),       //
                                            ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_CASE_P(smoke_Pooling_BatchN, PoolingLayerTest_NPU3700,
                        ::testing::Combine(pool_batchN,                                         //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{16, 16, 1, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                        PoolingLayerTest::getTestCaseName);

/* ============= Padding valitation ( > K_SZ/2) ============= */

const auto pool_LargePadding2 =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                              ::testing::ValuesIn<SizeVector>({{2, 2}, {3, 3}}),   // kernels
                                              ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                              ::testing::ValuesIn<SizeVector>({{2, 2}}),           // padBegins
                                              ::testing::ValuesIn<SizeVector>({{2, 2}}),           // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                              ::testing::Values(ngraph::op::PadType::VALID),       //
                                              ::testing::Values(false)),                           // excludePad
                           ::testing::Values(Precision::FP16),                                     // netPrc
                           ::testing::Values(Precision::FP16),                                     // inPrc
                           ::testing::Values(Precision::FP16),                                     // outPrc
                           ::testing::Values(Layout::ANY),                                         // inLayout
                           ::testing::Values(Layout::ANY),                                         // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),                     // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargePadding2, PoolingLayerTest_NPU3700, pool_LargePadding2,
                         PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding3 =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX),                       //
                                              ::testing::ValuesIn<SizeVector>({{3, 3}, {4, 4}, {5, 5}}),  // kernels
                                              ::testing::ValuesIn<SizeVector>({{1, 1}}),                  // strides
                                              ::testing::ValuesIn<SizeVector>({{3, 3}}),                  // padBegins
                                              ::testing::ValuesIn<SizeVector>({{3, 3}}),                  // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),         //
                                              ::testing::Values(ngraph::op::PadType::VALID),              //
                                              ::testing::Values(false)),                                  // excludePad
                           ::testing::Values(Precision::FP16),                                            // netPrc
                           ::testing::Values(Precision::FP16),                                            // inPrc
                           ::testing::Values(Precision::FP16),                                            // outPrc
                           ::testing::Values(Layout::ANY),                                                // inLayout
                           ::testing::Values(Layout::ANY),                                                // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),                            // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargePadding3, PoolingLayerTest_NPU3700, pool_LargePadding3,
                         PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding4 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                               //
                           ::testing::ValuesIn<SizeVector>({{4, 4}, {5, 5}, {6, 6}, {7, 7}}),  // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),                          // strides
                           ::testing::ValuesIn<SizeVector>({{4, 4}}),                          // padBegins
                           ::testing::ValuesIn<SizeVector>({{4, 4}}),                          // padEnds
                           ::testing::Values(ngraph::op::RoundingType::FLOOR),                 //
                           ::testing::Values(ngraph::op::PadType::VALID),                      //
                           ::testing::Values(false)),                                          // excludePad
        ::testing::Values(Precision::FP16),                                                    // netPrc
        ::testing::Values(Precision::FP16),                                                    // inPrc
        ::testing::Values(Precision::FP16),                                                    // outPrc
        ::testing::Values(Layout::ANY),                                                        // inLayout
        ::testing::Values(Layout::ANY),                                                        // outLayout
        ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),                                    // inputShapes
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargePadding4, PoolingLayerTest_NPU3700, pool_LargePadding4,
                         PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding5 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                       //
                           ::testing::ValuesIn<SizeVector>({{5, 5}, {6, 6}, {7, 7}, {8, 8}, {9, 9}}),  // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),                                  // strides
                           ::testing::ValuesIn<SizeVector>({{5, 5}}),                                  // padBegins
                           ::testing::ValuesIn<SizeVector>({{5, 5}}),                                  // padEnds
                           ::testing::Values(ngraph::op::RoundingType::FLOOR),                         //
                           ::testing::Values(ngraph::op::PadType::VALID),                              //
                           ::testing::Values(false)),                                                  // excludePad
        ::testing::Values(Precision::FP16),                                                            // netPrc
        ::testing::Values(Precision::FP16),                                                            // inPrc
        ::testing::Values(Precision::FP16),                                                            // outPrc
        ::testing::Values(Layout::ANY),                                                                // inLayout
        ::testing::Values(Layout::ANY),                                                                // outLayout
        ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),                                            // inputShapes
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargePadding5, PoolingLayerTest_NPU3700, pool_LargePadding5,
                         PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding6 = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(PoolingTypes::MAX),                                                   //
                ::testing::ValuesIn<SizeVector>({{6, 6}, {7, 7}, {8, 8}, {9, 9}, {10, 10}, {11, 11}}),  // kernels
                ::testing::ValuesIn<SizeVector>({{1, 1}}),                                              // strides
                ::testing::ValuesIn<SizeVector>({{6, 6}}),                                              // padBegins
                ::testing::ValuesIn<SizeVector>({{6, 6}}),                                              // padEnds
                ::testing::Values(ngraph::op::RoundingType::FLOOR),                                     //
                ::testing::Values(ngraph::op::PadType::VALID),                                          //
                ::testing::Values(false)),                                                              // excludePad
        ::testing::Values(Precision::FP16),                                                             // netPrc
        ::testing::Values(Precision::FP16),                                                             // inPrc
        ::testing::Values(Precision::FP16),                                                             // outPrc
        ::testing::Values(Layout::ANY),                                                                 // inLayout
        ::testing::Values(Layout::ANY),                                                                 // outLayout
        ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),                                             // inputShapes
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargePadding6, PoolingLayerTest_NPU3700, pool_LargePadding6,
                         PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding7 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                           //
                           ::testing::ValuesIn<SizeVector>({{7, 7}, {8, 8}, {9, 9}, {10, 10}, {11, 11}}),  // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),                                      // strides
                           ::testing::ValuesIn<SizeVector>({{7, 7}}),                                      // padBegins
                           ::testing::ValuesIn<SizeVector>({{7, 7}}),                                      // padEnds
                           ::testing::Values(ngraph::op::RoundingType::FLOOR),                             //
                           ::testing::Values(ngraph::op::PadType::VALID),                                  //
                           ::testing::Values(false)),                                                      // excludePad
        ::testing::Values(Precision::FP16),                                                                // netPrc
        ::testing::Values(Precision::FP16),                                                                // inPrc
        ::testing::Values(Precision::FP16),                                                                // outPrc
        ::testing::Values(Layout::ANY),                                                                    // inLayout
        ::testing::Values(Layout::ANY),                                                                    // outLayout
        ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargePadding7, PoolingLayerTest_NPU3700, pool_LargePadding7,
                         PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding8 = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                                   //
                           ::testing::ValuesIn<SizeVector>({{8, 8}, {9, 9}, {10, 10}, {11, 11}}),  // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),                              // strides
                           ::testing::ValuesIn<SizeVector>({{8, 8}}),                              // padBegins
                           ::testing::ValuesIn<SizeVector>({{8, 8}}),                              // padEnds
                           ::testing::Values(ngraph::op::RoundingType::FLOOR),                     //
                           ::testing::Values(ngraph::op::PadType::VALID),                          //
                           ::testing::Values(false)),                                              // excludePad
        ::testing::Values(Precision::FP16),                                                        // netPrc
        ::testing::Values(Precision::FP16),                                                        // inPrc
        ::testing::Values(Precision::FP16),                                                        // outPrc
        ::testing::Values(Layout::ANY),                                                            // inLayout
        ::testing::Values(Layout::ANY),                                                            // outLayout
        ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),                                        // inputShapes
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargePadding8, PoolingLayerTest_NPU3700, pool_LargePadding8,
                         PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large Kernels ============= */

const auto avgPool_largeKernels =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::AVG),                //
                                              ::testing::ValuesIn<SizeVector>({{23, 30}}),         // kernels
                                              ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                              ::testing::Values(ngraph::op::PadType::VALID),       //
                                              ::testing::Values(false)),                           // excludePad, //
                           ::testing::Values(Precision::FP16),                                     // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // outPrc
                           ::testing::Values(Layout::ANY),                                         // inLayout
                           ::testing::Values(Layout::ANY),                                         // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 2048, 23, 30}}),                   // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_AvgPooling_LargeKernels, PoolingLayerTest_NPU3700, avgPool_largeKernels,
                         PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large KernelsX ============= */

const auto avgPool_largeKernelsX =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::AVG),                //
                                              ::testing::ValuesIn<SizeVector>({{1, 14}}),          // kernels
                                              ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                              ::testing::Values(ngraph::op::PadType::VALID),       //
                                              ::testing::Values(false)),                           // excludePad
                           ::testing::Values(Precision::FP16),                                     // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // outPrc
                           ::testing::Values(Layout::ANY),                                         // inLayout
                           ::testing::Values(Layout::ANY),                                         // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 16, 1, 14}}),                      // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_AvgPooling_LargeKernelsX, PoolingLayerTest_NPU3700, avgPool_largeKernelsX,
                         PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large KernelsY ============= */

const auto avgPool_largeKernelsY =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::AVG),                //
                                              ::testing::ValuesIn<SizeVector>({{14, 1}}),          // kernels
                                              ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                              ::testing::Values(ngraph::op::PadType::VALID),       //
                                              ::testing::Values(false)),                           // excludePad,
                           ::testing::Values(Precision::FP16),                                     // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // outPrc
                           ::testing::Values(Layout::ANY),                                         // inLayout
                           ::testing::Values(Layout::ANY),                                         // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 16, 14, 1}}),                      // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_AvgPooling_LargeKernelsY, PoolingLayerTest_NPU3700, avgPool_largeKernelsY,
                         PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large Prime Kernels ============= */

const auto avgPool_largePrimeKernels =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::AVG),                //
                                              ::testing::ValuesIn<SizeVector>({{17, 17}}),         // kernels
                                              ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                              ::testing::Values(ngraph::op::PadType::VALID),       //
                                              ::testing::Values(false)),                           // excludePad,
                           ::testing::Values(Precision::FP16),                                     // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // outPrc
                           ::testing::Values(Layout::ANY),                                         // inLayout
                           ::testing::Values(Layout::ANY),                                         // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 147, 17, 17}}),                    // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_AvgPooling_LargePrimeKernels, PoolingLayerTest_NPU3700, avgPool_largePrimeKernels,
                         PoolingLayerTest::getTestCaseName);

/* ============= MAXPooling / Large Kernels ============= */

const auto maxPool_largeKernels =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                              ::testing::ValuesIn<SizeVector>({{23, 30}}),         // kernels
                                              ::testing::ValuesIn<SizeVector>({{23, 30}}),         // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                              ::testing::Values(ngraph::op::PadType::VALID),       //
                                              ::testing::Values(false)),                           // excludePad
                           ::testing::Values(Precision::FP16),                                     // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // outPrc
                           ::testing::Values(Layout::ANY),                                         // inLayout
                           ::testing::Values(Layout::ANY),                                         // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 2048, 23, 30}}),                   // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_MaxPooling_LargeKernels, PoolingLayerTest_NPU3700, maxPool_largeKernels,
                         PoolingLayerTest::getTestCaseName);

/* ============= MAXPooling / Large KernelsX ============= */

const auto maxPool_largeKernelsX =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                              ::testing::ValuesIn<SizeVector>({{1, 14}}),          // kernels
                                              ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                              ::testing::Values(ngraph::op::PadType::VALID),       //
                                              ::testing::Values(false)),                           // excludePad
                           ::testing::Values(Precision::FP16),                                     // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // outPrc
                           ::testing::Values(Layout::ANY),                                         // inLayout
                           ::testing::Values(Layout::ANY),                                         // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 16, 1, 14}}),                      // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_MaxPooling_LargeKernelsX, PoolingLayerTest_NPU3700, maxPool_largeKernelsX,
                         PoolingLayerTest::getTestCaseName);

/* ============= MAXPooling / Large KernelsY ============= */

const auto maxPool_largeKernelsY =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                              ::testing::ValuesIn<SizeVector>({{14, 1}}),          // kernels
                                              ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                              ::testing::Values(ngraph::op::PadType::VALID),       //
                                              ::testing::Values(false)),                           // excludePad
                           ::testing::Values(Precision::FP16),                                     // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // outPrc
                           ::testing::Values(Layout::ANY),                                         // inLayout
                           ::testing::Values(Layout::ANY),                                         // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 16, 14, 1}}),                      // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_MaxPooling_LargeKernelsY, PoolingLayerTest_NPU3700, maxPool_largeKernelsY,
                         PoolingLayerTest::getTestCaseName);

/* ============= AvgPooling / Exclude_Pad Handling ============= */

const auto avgPool_excludePad =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::AVG),                //
                                              ::testing::ValuesIn<SizeVector>({{3, 3}}),           // kernels
                                              ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                              ::testing::ValuesIn<SizeVector>({{1, 1}}),           // padBegins
                                              ::testing::ValuesIn<SizeVector>({{1, 1}}),           // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                              ::testing::Values(ngraph::op::PadType::VALID),       //
                                              ::testing::Values(true)),                            // excludePad,
                           ::testing::Values(Precision::FP16),                                     // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),                              // outPrc
                           ::testing::Values(Layout::ANY),                                         // inLayout
                           ::testing::Values(Layout::ANY),                                         // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 16, 28, 28}}),                     // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_avgPool_excludePad, PoolingLayerTest_NPU3700, avgPool_excludePad,
                         PoolingLayerTest::getTestCaseName);

/* ======================================== NPU 3720 ============================================================= */

/* ==== Custom tests scenario extra added for 3720 ===== */
const auto pool_ExplicitNoPadding_Params =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),
                                              ::testing::ValuesIn<SizeVector>({{14, 14}, {14, 1}, {1, 14}}),  // kernels
                                              ::testing::Values<SizeVector>({1, 1}),                          // strides
                                              ::testing::Values<SizeVector>({0, 0}),  // padBegins
                                              ::testing::Values<SizeVector>({0, 0}),  // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT),
                                              ::testing::Values(true)),     // excludePad
                           ::testing::Values(Precision::FP16),              // netPrc
                           ::testing::Values(Precision::FP16),              // inPrc
                           ::testing::Values(Precision::FP16),              // outPrc
                           ::testing::Values(Layout::ANY),                  // inLayout
                           ::testing::Values(Layout::ANY),                  // outLayout
                           ::testing::Values<SizeVector>({1, 30, 14, 14}),  // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// U-net usecase
const auto pool_unet_Params =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),
                                              ::testing::Values<SizeVector>({12, 1}),  // kernels
                                              ::testing::Values<SizeVector>({1, 1}),   // strides
                                              ::testing::Values<SizeVector>({0, 0}),   // padBegins
                                              ::testing::Values<SizeVector>({0, 0}),   // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT),
                                              ::testing::Values(true)),     // excludePad
                           ::testing::Values(Precision::FP16),              // netPrc
                           ::testing::Values(Precision::FP16),              // inPrc
                           ::testing::Values(Precision::FP16),              // outPrc
                           ::testing::Values(Layout::ANY),                  // inLayout
                           ::testing::Values(Layout::ANY),                  // outLayout
                           ::testing::Values<SizeVector>({1, 1, 12, 176}),  // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// large kernel
const auto pooling_largeKernel_Params =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),  //
                                              ::testing::ValuesIn<SizeVector>({{28, 28}}),              // kernels
                                              ::testing::ValuesIn<SizeVector>({{1, 1}}),                // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),                // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),                // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),       //
                                              ::testing::Values(ngraph::op::PadType::VALID),            //
                                              ::testing::Values(true)),                                 // excludePad
                           ::testing::Values(Precision::FP16),                                          // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),                                   // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),                                   // outPrc
                           ::testing::Values(Layout::ANY),                                              // inLayout
                           ::testing::Values(Layout::ANY),                                              // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 70, 28, 28}}),                          // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// Large kernel with stride 1
const auto pooling_largeKernelStrideOne =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                              ::testing::ValuesIn<SizeVector>({{71, 1}}),          // kernels
                                              ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                              ::testing::Values(ngraph::op::PadType::VALID),       //
                                              ::testing::Values(false)),      // excludePad,              //
                           ::testing::Values(Precision::FP16),                // netPrc
                           ::testing::Values(Precision::UNSPECIFIED),         // inPrc
                           ::testing::Values(Precision::UNSPECIFIED),         // outPrc
                           ::testing::Values(Layout::ANY),                    // inLayout
                           ::testing::Values(Layout::ANY),                    // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 1, 71, 2}}),  // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// Test all padding type
const auto poolAllPadTypeParams = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                           ::testing::ValuesIn<SizeVector>({{5, 7}}),  // kernels
                           ::testing::Values<SizeVector>({2, 3}),      // strides
                           ::testing::Values<SizeVector>({2, 3}),      // padBegins
                           ::testing::Values<SizeVector>({1, 2}),      // padEnds
                           ::testing::Values(ngraph::op::RoundingType::FLOOR, ngraph::op::RoundingType::CEIL),
                           ::testing::Values(ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::SAME_LOWER,
                                             ngraph::op::PadType::SAME_UPPER, ngraph::op::PadType::VALID),
                           ::testing::Values(true)),    // excludePad
        ::testing::Values(Precision::FP16),             // netPrc
        ::testing::Values(Precision::FP16),             // inPrc
        ::testing::Values(Precision::FP16),             // outPrc
        ::testing::Values(Layout::ANY),                 // inLayout
        ::testing::Values(Layout::ANY),                 // outLayout
        ::testing::Values<SizeVector>({1, 2, 30, 30}),  // inputShapes
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// 3D usecase
const auto pool3DParams = ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                                                                ::testing::ValuesIn<SizeVector>({{3}}),  // kernels
                                                                ::testing::ValuesIn<SizeVector>({{1}}),  // strides
                                                                ::testing::ValuesIn<SizeVector>({{1}}),  // padBegins
                                                                ::testing::ValuesIn<SizeVector>({{0}}),  // padEnds
                                                                ::testing::Values(ngraph::op::RoundingType::CEIL),
                                                                ::testing::Values(ngraph::op::PadType::SAME_UPPER),
                                                                ::testing::Values(false)),   // excludePad
                                             ::testing::Values(Precision::FP16),             // netPrc
                                             ::testing::Values(Precision::FP16),             // inPrc
                                             ::testing::Values(Precision::FP16),             // outPrc
                                             ::testing::Values(Layout::ANY),                 // inLayout
                                             ::testing::Values(Layout::ANY),                 // outLayout
                                             ::testing::ValuesIn<SizeVector>({{3, 4, 64}}),  // inputShapes
                                             ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// 5d usecase
const auto pool5DParams =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                                              ::testing::ValuesIn<SizeVector>({{2, 2, 2}}),  // kernels
                                              ::testing::ValuesIn<SizeVector>({{2, 2, 2}}),  // strides
                                              ::testing::ValuesIn<SizeVector>({{0, 0, 0}}),  // padBegins
                                              ::testing::ValuesIn<SizeVector>({{0, 0, 0}}),  // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),
                                              ::testing::Values(ngraph::op::PadType::SAME_UPPER),
                                              ::testing::Values(true)),           // excludePad
                           ::testing::Values(Precision::FP16),                    // netPrc
                           ::testing::Values(Precision::FP16),                    // inPrc
                           ::testing::Values(Precision::FP16),                    // outPrc
                           ::testing::Values(Layout::ANY),                        // inLayout
                           ::testing::Values(Layout::ANY),                        // outLayout
                           ::testing::ValuesIn<SizeVector>({{1, 4, 16, 8, 12}}),  // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// pad outside of kernel size/2. Pad is valid until at kerneSize-1.
const auto pooligBigPadEndParams = ::testing::Combine(
        ::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                           ::testing::ValuesIn<SizeVector>({{3, 3}}),  // kernels
                           ::testing::Values<SizeVector>({2, 2}),      // strides
                           ::testing::Values<SizeVector>({0, 0}),      // padBegins
                           ::testing::Values<SizeVector>({2, 2}),      // padEnds
                           ::testing::Values(ngraph::op::RoundingType::FLOOR, ngraph::op::RoundingType::CEIL),
                           ::testing::Values(ngraph::op::PadType::EXPLICIT),
                           ::testing::Values(false)),   // excludePad
        ::testing::Values(Precision::FP16),             // netPrc
        ::testing::Values(Precision::FP16),             // inPrc
        ::testing::Values(Precision::FP16),             // outPrc
        ::testing::Values(Layout::ANY),                 // inLayout
        ::testing::Values(Layout::ANY),                 // outLayout
        ::testing::Values<SizeVector>({1, 4, 54, 54}),  // inputShapes
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

// basic usecase
const auto pool_basic_Params =
        ::testing::Combine(::testing::Combine(::testing::Values(PoolingTypes::AVG, PoolingTypes::MAX),
                                              ::testing::Values<SizeVector>({3, 3}),  // kernels
                                              ::testing::Values<SizeVector>({1, 1}),  // strides
                                              ::testing::Values<SizeVector>({1, 1}),  // padBegins
                                              ::testing::Values<SizeVector>({1, 1}),  // padEnds
                                              ::testing::Values(ngraph::op::RoundingType::FLOOR),
                                              ::testing::Values(ngraph::op::PadType::EXPLICIT),
                                              ::testing::Values(false)),   // excludePad
                           ::testing::Values(Precision::FP32),             // netPrc
                           ::testing::Values(Precision::FP32),             // inPrc
                           ::testing::Values(Precision::FP32),             // outPrc
                           ::testing::Values(Layout::ANY),                 // inLayout
                           ::testing::Values(Layout::ANY),                 // outLayout
                           ::testing::Values<SizeVector>({1, 2, 16, 24}),  // inputShapes
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_NCHW_NoPadding, PoolingLayerTest_NPU3720, pool_ExplicitNoPadding_Params,
                         PoolingLayerTest_NPU3720::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_NCHW_NoPadding_ELF, PoolingLayerTest_NPU3720_SingleCluster,
                         pool_ExplicitNoPadding_Params, PoolingLayerTest_NPU3720::getTestCaseName);
// U-net usecase
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Pooling_unet, PoolingLayerTest_NPU3720, pool_unet_Params,
                         PoolingLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_unet_ELF, PoolingLayerTest_NPU3720_SingleCluster, pool_unet_Params,
                         PoolingLayerTest_NPU3720::getTestCaseName);
// large kernel
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargeKernel, PoolingLayerTest_NPU3720, pooling_largeKernel_Params,
                         PoolingLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_LargeKernel3720_ELF, PoolingLayerTest_NPU3720_SingleCluster,
                         pooling_largeKernel_Params, PoolingLayerTest_NPU3720::getTestCaseName);
// Large kernel with stride 1
INSTANTIATE_TEST_SUITE_P(smoke_precommit_Pooling_LargeKernelStrideOne, PoolingLayerTest_NPU3720,
                         pooling_largeKernelStrideOne, PoolingLayerTest_NPU3720::getTestCaseName);
// all PadType
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AllPadType, PoolingLayerTest_NPU3720, poolAllPadTypeParams,
                         PoolingLayerTest::getTestCaseName);
// 3D usecase
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_3D, PoolingLayerTest_NPU3720, pool3DParams,
                         PoolingLayerTest_NPU3720::getTestCaseName);
// 5d usecase
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_5D, PoolingLayerTest_NPU3720, pool5DParams,
                         PoolingLayerTest_NPU3720::getTestCaseName);
// pad outside of kernel size/2. Pad is valid until at kerneSize-1.
INSTANTIATE_TEST_SUITE_P(smoke_Pooling_BigPadEndParams, PoolingLayerTest_NPU3720, pooligBigPadEndParams,
                         PoolingLayerTest_NPU3720::getTestCaseName);
// previous cip reused tests
/* ============= AutoPadValid ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AutoPadValid, PoolingLayerTest_NPU3720, pool_AutoPadValid,
                                             PoolingLayerTest::getTestCaseName);
/* ============= ExplicitPadding ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_ExplicitPadding, PoolingLayerTest_NPU3720,
                                             pool_ExplicitPadding, PoolingLayerTest::getTestCaseName);
/* ============= AsymmetricKernel ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AsymmetricKernel, PoolingLayerTest_NPU3720,
                                             pool_AsymmetricKernel, PoolingLayerTest::getTestCaseName);
/* ============= AsymmetricStrides ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_AsymmetricStrides, PoolingLayerTest_NPU3720,
                                             pool_AsymmetricStrides, PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large Kernels ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernels, PoolingLayerTest_NPU3720,
                                             avgPool_largeKernels, PoolingLayerTest::getTestCaseName);
/* ============= AVGPooling / Large KernelsX ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernelsX, PoolingLayerTest_NPU3720,
                                             avgPool_largeKernelsX, PoolingLayerTest::getTestCaseName);
/* ============= AVGPooling / Large KernelsY ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargeKernelsY, PoolingLayerTest_NPU3720,
                                             avgPool_largeKernelsY, PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large Prime Kernels ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_AvgPooling_LargePrimeKernels, PoolingLayerTest_NPU3720,
                                             avgPool_largePrimeKernels, PoolingLayerTest::getTestCaseName);
/* ============= AvgPooling / Exclude_Pad Handling ============= */
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_avgPool_excludePad, PoolingLayerTest_NPU3720, avgPool_excludePad,
                                             PoolingLayerTest::getTestCaseName);

// Max pool ported tests from VPU3700
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeSize1, PoolingLayerTest_NPU3720, pool_LargeSize1,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeSize2, PoolingLayerTest_NPU3720, pool_LargeSize2,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargeStrides, PoolingLayerTest_NPU3720, pool_LargeStrides,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding2, PoolingLayerTest_NPU3720, pool_LargePadding2,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding3, PoolingLayerTest_NPU3720, pool_LargePadding3,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding4, PoolingLayerTest_NPU3720, pool_LargePadding4,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding5, PoolingLayerTest_NPU3720, pool_LargePadding5,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding6, PoolingLayerTest_NPU3720, pool_LargePadding6,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding7, PoolingLayerTest_NPU3720, pool_LargePadding7,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_Pooling_LargePadding8, PoolingLayerTest_NPU3720, pool_LargePadding8,
                                             PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernels, PoolingLayerTest_NPU3720,
                                             maxPool_largeKernels, PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernelsX, PoolingLayerTest_NPU3720,
                                             maxPool_largeKernelsX, PoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P_WITH_DISABLE_OPTION(smoke_MaxPooling_LargeKernelsY, PoolingLayerTest_NPU3720,
                                             maxPool_largeKernelsY, PoolingLayerTest::getTestCaseName);

}  // namespace
