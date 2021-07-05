// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/pooling.hpp"

#include <vector>

#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbPoolingLayerTest : public PoolingLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeLoad() override {
        const auto& poolParams = std::get<0>(GetParam());

        ngraph::helpers::PoolingTypes poolType;
        std::vector<size_t> strides;
        ngraph::op::RoundingType roundingMode;
        std::tie(poolType, std::ignore, strides, std::ignore, std::ignore, roundingMode, std::ignore, std::ignore) =
                poolParams;

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

            if (poolType == ngraph::helpers::PoolingTypes::AVG && roundingMode == ngraph::op::RoundingType::CEIL) {
                throw LayerTestsUtils::KmbSkipTestException("MCM compiler issues with AVG pool & CEIL rounding mode");
            }
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
    setReferenceHardwareModeMLIR();
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

INSTANTIATE_TEST_CASE_P(smoke_Pooling_AutoPadValid, KmbPoolingLayerTest,
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

INSTANTIATE_TEST_CASE_P(smoke_Pooling_ExplicitPadding, KmbPoolingLayerTest,
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

INSTANTIATE_TEST_CASE_P(smoke_Pooling_AsymmetricKernel, KmbPoolingLayerTest,
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

INSTANTIATE_TEST_CASE_P(smoke_Pooling_AsymmetricStrides, KmbPoolingLayerTest,
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

}  // namespace
