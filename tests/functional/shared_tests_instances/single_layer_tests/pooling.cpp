// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/pooling.hpp"
#include "vpux_private_config.hpp"

#include <vector>

#include "kmb_layer_test.hpp"
#include <common/functions.h>

namespace LayerTestsDefinitions {

class KmbPoolingLayerTest : public PoolingLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        const auto& poolParams = std::get<0>(GetParam());

        ngraph::helpers::PoolingTypes poolType;
        std::vector<size_t> strides;
        ngraph::op::RoundingType roundingMode;
        std::tie(poolType, std::ignore, strides, std::ignore, std::ignore, roundingMode, std::ignore, std::ignore) =
                poolParams;

        if (poolType == ngraph::helpers::PoolingTypes::AVG && isCompilerMLIR() &&
            configuration[VPUX_CONFIG_KEY(COMPILATION_MODE)] == "DefaultHW") {
            threshold = 0.25;
        }

        if (isCompilerMLIR()) {
            // MLIR uses software layer, which seem to be flawed
            // MCM uses hardware implementation of AvgPool, replacing with DW Conv
            if (poolType == ngraph::helpers::PoolingTypes::AVG) {
                if (strides[0] != 1 || strides[1] != 1) {
                    throw LayerTestsUtils::KmbSkipTestException("AVG pool strides != 1 produces inaccurate results");
                }
            }
        } else {
            if (strides[0] != strides[1]) {
                throw LayerTestsUtils::KmbSkipTestException("MCM compiler issues with asymmetric strides");
            }

            if (strides[0] > 8 || strides[1] > 8) {
                throw LayerTestsUtils::KmbSkipTestException("MCM compiler issues with large strides");
            }

            if (poolType == ngraph::helpers::PoolingTypes::AVG && roundingMode == ngraph::op::RoundingType::CEIL) {
                throw LayerTestsUtils::KmbSkipTestException("MCM compiler issues with AVG pool & CEIL rounding mode");
            }
        }
    }

    void SkipBeforeInfer() override {
        const auto& poolParams = std::get<0>(GetParam());

        std::vector<size_t> strides;
        std::tie(std::ignore, std::ignore, strides, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) =
                poolParams;
        // [Track number: E#20948]
        const auto testName =
            std::string{::testing::UnitTest::GetInstance()->current_test_info()->test_case_name()};
        const auto isSmokePoolAutoPadVal = testName.find("smoke_Pooling_AutoPadValid") != std::string::npos;
        const auto isLevel0 = getBackendName(*getCore()) == "LEVEL0";
        const auto failedStrides = strides.size() == 2 && strides[0] == 1 && strides[1] == 1;
        if (isSmokePoolAutoPadVal && isLevel0 && isCompilerMCM() && failedStrides) {
            throw LayerTestsUtils::KmbSkipTestException("Level0: sporadic failure on device");
        }
    }
};

TEST_P(KmbPoolingLayerTest, CompareWithRefs_MCM) {
    Run();
}

TEST_P(KmbPoolingLayerTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbPoolingLayerTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace InferenceEngine;
using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;

namespace {

/* ============= AutoPadValid ============= */

const auto pool_AutoPadValid = ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),  //
                                                  ::testing::ValuesIn<SizeVector>({{3, 3}, {5, 5}}),        // kernels
                                                  ::testing::ValuesIn<SizeVector>({{1, 1}, {2, 2}}),        // strides
                                                  ::testing::ValuesIn<SizeVector>({{0, 0}}),                // padBegins
                                                  ::testing::ValuesIn<SizeVector>({{0, 0}}),                // padEnds
                                                  ::testing::Values(ngraph::op::RoundingType::FLOOR),       //
                                                  ::testing::Values(ngraph::op::PadType::VALID),            //
                                                  ::testing::Values(false)  // excludePad
);

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AutoPadValid, KmbPoolingLayerTest,
                        ::testing::Combine(pool_AutoPadValid,                          //
                                           ::testing::Values(Precision::FP16),         // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),  // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),  // outPrc
                                           ::testing::Values(Layout::ANY),             // inLayout
                                           ::testing::Values(Layout::ANY),             // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 8, 32, 32},
                                                                            {1, 16, 24, 24},
                                                                            {1, 24, 16, 16},
                                                                            {1, 32, 8, 8}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        PoolingLayerTest::getTestCaseName);

/* ============= ExplicitPadding ============= */

const auto pool_ExplicitPadding =
        ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),    //
                           ::testing::ValuesIn<SizeVector>({{3, 3}}),                  // kernels
                           ::testing::ValuesIn<SizeVector>({{2, 2}}),                  // strides
                           ::testing::ValuesIn<SizeVector>({{0, 0}, {1, 1}, {0, 1}}),  // padBegins
                           ::testing::ValuesIn<SizeVector>({{0, 0}, {1, 1}, {0, 1}}),  // padEnds
                           ::testing::Values(ngraph::op::RoundingType::FLOOR, ngraph::op::RoundingType::CEIL),  //
                           ::testing::Values(ngraph::op::PadType::EXPLICIT),                                    //
                           ::testing::Values(false)  // excludePad
        );

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_ExplicitPadding, KmbPoolingLayerTest,
                        ::testing::Combine(pool_ExplicitPadding,                                //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),           // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),           // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 30, 30}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        PoolingLayerTest::getTestCaseName);

/* ============= AsymmetricKernel ============= */

const auto pool_AsymmetricKernel = ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),  //
                                                      ::testing::ValuesIn<SizeVector>({{3, 1}, {1, 3}}),   // kernels
                                                      ::testing::ValuesIn<SizeVector>({{1, 1}, {2, 2}}),   // strides
                                                      ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                                      ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                                      ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                      ::testing::Values(ngraph::op::PadType::VALID),       //
                                                      ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AsymmetricKernel, KmbPoolingLayerTest,
                        ::testing::Combine(pool_AsymmetricKernel,                               //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),           // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),           // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 30, 30}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        PoolingLayerTest::getTestCaseName);

/* ============= AsymmetricStrides ============= */

const auto pool_AsymmetricStrides = ::testing::Combine(::testing::Values(PoolingTypes::MAX, PoolingTypes::AVG),  //
                                                       ::testing::ValuesIn<SizeVector>({{3, 3}}),           // kernels
                                                       ::testing::ValuesIn<SizeVector>({{1, 2}, {2, 1}}),   // strides
                                                       ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                                       ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                                       ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                       ::testing::Values(ngraph::op::PadType::VALID),       //
                                                       ::testing::Values(false)  // excludePad
);

INSTANTIATE_TEST_SUITE_P(smoke_Pooling_AsymmetricStrides, KmbPoolingLayerTest,
                        ::testing::Combine(pool_AsymmetricStrides,                              //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),           // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),           // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 30, 30}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        PoolingLayerTest::getTestCaseName);

/* ============= LargeSize ============= */

const auto pool_LargeSize1 = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                                ::testing::ValuesIn<SizeVector>({{3, 3}}),           // kernels
                                                ::testing::ValuesIn<SizeVector>({{2, 2}}),           // strides
                                                ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                                ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                                ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                ::testing::Values(ngraph::op::PadType::VALID),       //
                                                ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_CASE_P(smoke_Pooling_LargeSize1, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargeSize1,                                       //
                                           ::testing::Values(Precision::FP16),                    // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // outPrc
                                           ::testing::Values(Layout::ANY),                        // inLayout
                                           ::testing::Values(Layout::ANY),                        // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 64, 128, 128}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        PoolingLayerTest::getTestCaseName);

const auto pool_LargeSize2 = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                //
                                                ::testing::ValuesIn<SizeVector>({{3, 3}}),           // kernels
                                                ::testing::ValuesIn<SizeVector>({{2, 2}}),           // strides
                                                ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                                                ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                                                ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                                                ::testing::Values(ngraph::op::PadType::VALID),       //
                                                ::testing::Values(false)                             // excludePad
);

INSTANTIATE_TEST_CASE_P(smoke_Pooling_LargeSize2, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargeSize2,                                       //
                                           ::testing::Values(Precision::FP16),                    // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // outPrc
                                           ::testing::Values(Layout::ANY),                        // inLayout
                                           ::testing::Values(Layout::ANY),                        // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 256, 256}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        PoolingLayerTest::getTestCaseName);

/* ============= LargeStrides ============= */

const auto pool_LargeStrides = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                 //
                                                  ::testing::ValuesIn<SizeVector>({{3, 3}, {11, 11}}),  // kernels
                                                  ::testing::ValuesIn<SizeVector>({{9, 9}}),            // strides
                                                  ::testing::ValuesIn<SizeVector>({{0, 0}}),            // padBegins
                                                  ::testing::ValuesIn<SizeVector>({{0, 0}}),            // padEnds
                                                  ::testing::Values(ngraph::op::RoundingType::FLOOR),   //
                                                  ::testing::Values(ngraph::op::PadType::VALID),        //
                                                  ::testing::Values(false)                              // excludePad
);

INSTANTIATE_TEST_CASE_P(smoke_Pooling_LargeStrides, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargeStrides,                                   //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        PoolingLayerTest::getTestCaseName);

/* ============= Padding valitation ( > K_SZ/2) ============= */

const auto pool_LargePadding2 = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                 //
                                                  ::testing::ValuesIn<SizeVector>({{2, 2}, {3,3}}),  // kernels
                                                  ::testing::ValuesIn<SizeVector>({{1, 1}}),            // strides
                                                  ::testing::ValuesIn<SizeVector>({{2, 2}}),            // padBegins
                                                  ::testing::ValuesIn<SizeVector>({{2, 2}}),            // padEnds
                                                  ::testing::Values(ngraph::op::RoundingType::FLOOR),   //
                                                  ::testing::Values(ngraph::op::PadType::VALID),        //
                                                  ::testing::Values(false)                              // excludePad
);

INSTANTIATE_TEST_CASE_P(smoke_Pooling_LargePadding2, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargePadding2,                                   //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        PoolingLayerTest::getTestCaseName);

const auto pool_LargePadding3 = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                 //
                                                  ::testing::ValuesIn<SizeVector>({{3,3}, {4,4}, {5, 5}}),  // kernels
                                                  ::testing::ValuesIn<SizeVector>({{1, 1}}),            // strides
                                                  ::testing::ValuesIn<SizeVector>({{3, 3}}),            // padBegins
                                                  ::testing::ValuesIn<SizeVector>({{3, 3}}),            // padEnds
                                                  ::testing::Values(ngraph::op::RoundingType::FLOOR),   //
                                                  ::testing::Values(ngraph::op::PadType::VALID),        //
                                                  ::testing::Values(false)                              // excludePad
);

INSTANTIATE_TEST_CASE_P(smoke_Pooling_LargePadding3, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargePadding3,                                   //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        PoolingLayerTest::getTestCaseName);


const auto pool_LargePadding4 = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                 //
                                                  ::testing::ValuesIn<SizeVector>({{4,4}, {5, 5}, {6,6}, {7,7}}),  // kernels
                                                  ::testing::ValuesIn<SizeVector>({{1, 1}}),            // strides
                                                  ::testing::ValuesIn<SizeVector>({{4, 4}}),            // padBegins
                                                  ::testing::ValuesIn<SizeVector>({{4, 4}}),            // padEnds
                                                  ::testing::Values(ngraph::op::RoundingType::FLOOR),   //
                                                  ::testing::Values(ngraph::op::PadType::VALID),        //
                                                  ::testing::Values(false)                              // excludePad
);

INSTANTIATE_TEST_CASE_P(smoke_Pooling_LargePadding4, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargePadding4,                                   //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        PoolingLayerTest::getTestCaseName);


const auto pool_LargePadding5 = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                 //
                                                  ::testing::ValuesIn<SizeVector>({{5, 5}, {6,6}, {7,7}, {8,8}, {9,9}}),  // kernels
                                                  ::testing::ValuesIn<SizeVector>({{1, 1}}),            // strides
                                                  ::testing::ValuesIn<SizeVector>({{5, 5}}),            // padBegins
                                                  ::testing::ValuesIn<SizeVector>({{5, 5}}),            // padEnds
                                                  ::testing::Values(ngraph::op::RoundingType::FLOOR),   //
                                                  ::testing::Values(ngraph::op::PadType::VALID),        //
                                                  ::testing::Values(false)                              // excludePad
);

INSTANTIATE_TEST_CASE_P(smoke_Pooling_LargePadding5, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargePadding5,                                   //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        PoolingLayerTest::getTestCaseName);


const auto pool_LargePadding6 = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                 //
                                                  ::testing::ValuesIn<SizeVector>({{6,6}, {7,7}, {8,8}, {9,9}, {10,10}, {11,11}}),  // kernels
                                                  ::testing::ValuesIn<SizeVector>({{1, 1}}),            // strides
                                                  ::testing::ValuesIn<SizeVector>({{6, 6}}),            // padBegins
                                                  ::testing::ValuesIn<SizeVector>({{6, 6}}),            // padEnds
                                                  ::testing::Values(ngraph::op::RoundingType::FLOOR),   //
                                                  ::testing::Values(ngraph::op::PadType::VALID),        //
                                                  ::testing::Values(false)                              // excludePad
);

INSTANTIATE_TEST_CASE_P(smoke_Pooling_LargePadding6, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargePadding6,                                   //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        PoolingLayerTest::getTestCaseName);


const auto pool_LargePadding7 = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                 //
                                                  ::testing::ValuesIn<SizeVector>({{7,7}, {8,8}, {9,9}, {10,10}, {11,11}}),  // kernels
                                                  ::testing::ValuesIn<SizeVector>({{1, 1}}),            // strides
                                                  ::testing::ValuesIn<SizeVector>({{7, 7}}),            // padBegins
                                                  ::testing::ValuesIn<SizeVector>({{7, 7}}),            // padEnds
                                                  ::testing::Values(ngraph::op::RoundingType::FLOOR),   //
                                                  ::testing::Values(ngraph::op::PadType::VALID),        //
                                                  ::testing::Values(false)                              // excludePad
);

INSTANTIATE_TEST_CASE_P(smoke_Pooling_LargePadding7, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargePadding7,                                   //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        PoolingLayerTest::getTestCaseName);


const auto pool_LargePadding8 = ::testing::Combine(::testing::Values(PoolingTypes::MAX),                 //
                                                  ::testing::ValuesIn<SizeVector>({{8,8}, {9,9}, {10,10}, {11,11}}),  // kernels
                                                  ::testing::ValuesIn<SizeVector>({{1, 1}}),            // strides
                                                  ::testing::ValuesIn<SizeVector>({{8, 8}}),            // padBegins
                                                  ::testing::ValuesIn<SizeVector>({{8, 8}}),            // padEnds
                                                  ::testing::Values(ngraph::op::RoundingType::FLOOR),   //
                                                  ::testing::Values(ngraph::op::PadType::VALID),        //
                                                  ::testing::Values(false)                              // excludePad
);

INSTANTIATE_TEST_CASE_P(smoke_Pooling_LargePadding8, KmbPoolingLayerTest,
                        ::testing::Combine(pool_LargePadding8,                                   //
                                           ::testing::Values(Precision::FP16),                  // netPrc
                                           ::testing::Values(Precision::FP16),                  // inPrc
                                           ::testing::Values(Precision::FP16),                  // outPrc
                                           ::testing::Values(Layout::ANY),                      // inLayout
                                           ::testing::Values(Layout::ANY),                      // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large Kernels ============= */

const auto avgPool_largeKernels =
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                 //
                            ::testing::ValuesIn<SizeVector>({{23, 30}}),         // kernels
                            ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                            ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                            ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                            ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                            ::testing::Values(ngraph::op::PadType::VALID),       //
                            ::testing::Values(false)                             // excludePad
        );

INSTANTIATE_TEST_CASE_P(smoke_AvgPooling_LargeKernels, KmbPoolingLayerTest,
                        ::testing::Combine(avgPool_largeKernels,                               //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 2048, 23, 30}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large KernelsX ============= */

const auto avgPool_largeKernelsX =
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                 //
                            ::testing::ValuesIn<SizeVector>({{1, 14}}),         // kernels
                            ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                            ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                            ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                            ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                            ::testing::Values(ngraph::op::PadType::VALID),       //
                            ::testing::Values(false)                             // excludePad
        );

INSTANTIATE_TEST_CASE_P(smoke_AvgPooling_LargeKernelsX, KmbPoolingLayerTest,
                        ::testing::Combine(avgPool_largeKernelsX,                               //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 1, 14}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= AVGPooling / Large KernelsY ============= */

const auto avgPool_largeKernelsY =
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                 //
                            ::testing::ValuesIn<SizeVector>({{14, 1}}),         // kernels
                            ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                            ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                            ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                            ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                            ::testing::Values(ngraph::op::PadType::VALID),       //
                            ::testing::Values(false)                             // excludePad
        );

INSTANTIATE_TEST_CASE_P(smoke_AvgPooling_LargeKernelsY, KmbPoolingLayerTest,
                        ::testing::Combine(avgPool_largeKernelsY,                               //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 14, 1}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= MAXPooling / Large Kernels ============= */

const auto maxPool_largeKernels =
        ::testing::Combine(::testing::Values(PoolingTypes::MAX),                 //
                            ::testing::ValuesIn<SizeVector>({{23, 30}}),         // kernels
                            ::testing::ValuesIn<SizeVector>({{23, 30}}),           // strides
                            ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                            ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                            ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                            ::testing::Values(ngraph::op::PadType::VALID),       //
                            ::testing::Values(false)                             // excludePad
        );

INSTANTIATE_TEST_CASE_P(smoke_MaxPooling_LargeKernels, KmbPoolingLayerTest,
                        ::testing::Combine(maxPool_largeKernels,                     //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 2048, 23, 30}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= MAXPooling / Large KernelsX ============= */

const auto maxPool_largeKernelsX =
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                 //
                            ::testing::ValuesIn<SizeVector>({{1, 14}}),         // kernels
                            ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                            ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                            ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                            ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                            ::testing::Values(ngraph::op::PadType::VALID),       //
                            ::testing::Values(false)                             // excludePad
        );

INSTANTIATE_TEST_CASE_P(smoke_MaxPooling_LargeKernelsX, KmbPoolingLayerTest,
                        ::testing::Combine(maxPool_largeKernelsX,                               //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 1, 14}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);

/* ============= MAXPooling / Large KernelsY ============= */

const auto maxPool_largeKernelsY =
        ::testing::Combine(::testing::Values(PoolingTypes::AVG),                 //
                            ::testing::ValuesIn<SizeVector>({{14, 1}}),         // kernels
                            ::testing::ValuesIn<SizeVector>({{1, 1}}),           // strides
                            ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padBegins
                            ::testing::ValuesIn<SizeVector>({{0, 0}}),           // padEnds
                            ::testing::Values(ngraph::op::RoundingType::FLOOR),  //
                            ::testing::Values(ngraph::op::PadType::VALID),       //
                            ::testing::Values(false)                             // excludePad
        );

INSTANTIATE_TEST_CASE_P(smoke_MaxPooling_LargeKernelsY, KmbPoolingLayerTest,
                        ::testing::Combine(maxPool_largeKernelsY,                               //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 14, 1}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        PoolingLayerTest::getTestCaseName);


/* ============= Adaptive_AVG_Pool / 3D ============= */

     const std::vector<InferenceEngine::Precision> inputPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16,
            InferenceEngine::Precision::U8,
    };


const auto AdaPool3DCases =
        ::testing::Combine(::testing::ValuesIn(
                std::vector<std::vector<size_t>> {
                        { 1, 2, 1},
                        { 1, 1, 3 },
                        { 3, 17, 5 }}),
        ::testing::ValuesIn(std::vector<std::vector<int>>{ {1}, {3}, {5} }),
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_TestsAdaPool3D, KmbPoolingLayerTest, AdaPool3DCases, PoolingLayerTest::getTestCaseName);

/* ============= Adaptive_AVG_Pool / 3D ============= */
/*
const auto AdaPool4DCases = ::testing::Combine(
        ::testing::ValuesIn(
                std::vector<std::vector<size_t>> {
                        { 1, 2, 1, 2},
                        { 1, 1, 3, 2},
                        { 3, 17, 5, 1}}),
        ::testing::ValuesIn(std::vector<std::vector<int>>{ {1, 1}, {3, 5}, {5, 5} }),
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_TestsAdaPool4D, KmbPoolingLayerTest, AdaPool4DCases, PoolingLayerTest::getTestCaseName);

/* ============= Adaptive_AVG_Pool / 3D ============= */
/*
const auto AdaPool5DCases = ::testing::Combine(
        ::testing::ValuesIn(
                std::vector<std::vector<size_t>> {
                        { 1, 2, 1, 2, 2},
                        { 1, 1, 3, 2, 3},
                        { 3, 17, 5, 1, 2}}),
        ::testing::ValuesIn(std::vector<std::vector<int>>{ {1, 1, 1}, {3, 5, 3}, {5, 5, 5} }),
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

INSTANTIATE_TEST_CASE_P(smoke_TestsAdaPool5D, KmbPoolingLayerTest, AdaPool5DCases, PoolingLayerTest::getTestCaseName);
*/
}  // namespace
