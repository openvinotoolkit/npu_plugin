// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/convolution.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"

namespace LayerTestsDefinitions {

class KmbConvolutionLayerTest : public ConvolutionLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
    void SkipBeforeValidate() override {
        if (isCompilerMLIR()) {
            auto params = std::get<0>(GetParam());

            auto strides = std::get<1>(params);
            auto padsBegin = std::get<2>(params);
            auto dilations = std::get<4>(params);
            auto padType = std::get<6>(params);

            // [Track number: E#16206]
            if (strides.size() == 1 && padType == ngraph::op::PadType::EXPLICIT) {
                auto isBadPadsBegin = padsBegin[0] > 1;
                auto isBadDilations = dilations[0] == 1;
                if (isBadPadsBegin && isBadDilations) {
                    throw LayerTestsUtils::KmbSkipTestException("Comparison fails");
                }
            }

            // [Track number: E#16206]
            if (strides.size() == 2 && padType == ngraph::op::PadType::EXPLICIT) {
                auto isBadStrides = strides[0] == strides[1] && strides[0] < 5;
                auto isGoodPadsBegin =
                        (padsBegin[0] == 0 && padsBegin[1] == 0) || (padsBegin[0] == 1 && padsBegin[1] == 0) ||
                        (padsBegin[0] == 0 && padsBegin[1] == 1) || (padsBegin[0] == 1 && padsBegin[1] == 1);
                auto isBadDilations = dilations[0] == 1 && dilations[1] == 1;

                if (isBadStrides && !isGoodPadsBegin && isBadDilations)
                    throw LayerTestsUtils::KmbSkipTestException("Comparison fails");
            }
        }
    }
};

// Comparisons fail (ticket???)
TEST_P(KmbConvolutionLayerTest, DISABLED_CompareWithRefs) {
    Run();
}

TEST_P(KmbConvolutionLayerTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbConvolutionLayerTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setReferenceHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace InferenceEngine;
using namespace LayerTestsDefinitions;

namespace {

/* ============= 1D Convolution ============= */

const auto conv1DParams = ::testing::Combine(::testing::ValuesIn<SizeVector>({{1}, {5}}),              // kernels
                                             ::testing::ValuesIn<SizeVector>({{1}, {3}}),              // strides
                                             ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0}, {3}}),  // padBegins
                                             ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0}, {2}}),  // padEnds
                                             ::testing::ValuesIn<SizeVector>({{1}, {2}}),              // dilations
                                             ::testing::Values(1, 4),                                  // numOutChannels
                                             ::testing::Values(ngraph::op::PadType::EXPLICIT)          // padType
);

INSTANTIATE_TEST_CASE_P(smoke_Convolution1D, KmbConvolutionLayerTest,
                        ::testing::Combine(conv1DParams,
                                           ::testing::Values(Precision::FP16),              // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),       // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),       // outPrc
                                           ::testing::Values(Layout::ANY),                  // inLayout
                                           ::testing::Values(Layout::ANY),                  // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 64}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / AutoPadValid ============= */

const auto conv2DParams_AutoPadValid =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{1, 1}, {3, 3}}),      // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}, {2, 2}}),      // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                           ::testing::Values(8, 16, 24, 32),                       // numOutChannels
                           ::testing::Values(ngraph::op::PadType::VALID)           // padType
        );

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_AutoPadValid, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_AutoPadValid,                  //
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
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / ExplicitPadding ============= */

const auto conv2DParams_ExplicitPadding =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{3, 3}}),                              // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}, {2, 2}}),                              // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}, {1, 1}, {0, 1}, {0, 2}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}, {1, 1}, {0, 1}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),                              // dilations
                           ::testing::Values(1),                                                   // numOutChannels
                           ::testing::Values(ngraph::op::PadType::EXPLICIT)                        // padType
        );

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_ExplicitPadding, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_ExplicitPadding,                       //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 3, 16, 16}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / AsymmetricKernel ============= */

const auto conv2DParams_AsymmetricKernel =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{3, 1}, {1, 3}}),      // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}, {2, 2}}),      // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                           ::testing::Values(1),                                   // numOutChannels
                           ::testing::Values(ngraph::op::PadType::VALID)           // padType
        );

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_AsymmetricKernel, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_AsymmetricKernel,                      //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 3, 16, 16}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / AsymmetricStrides ============= */

const auto conv2DParams_AsymmetricStrides =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{3, 3}}),              // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 2}, {2, 1}}),      // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                           ::testing::Values(1),                                   // numOutChannels
                           ::testing::Values(ngraph::op::PadType::VALID)           // padType
        );

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_AsymmetricStrides, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_AsymmetricStrides,                     //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 3, 16, 16}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / Dilated ============= */

const auto conv2DParams_Dilated =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{3, 3}}),              // kernels
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{2, 2}}),              // dilations
                           ::testing::Values(1),                                   // numOutChannels
                           ::testing::Values(ngraph::op::PadType::VALID)           // padType
        );

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_Dilated, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_Dilated,                               //
                                           ::testing::Values(Precision::FP16),                 // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),          // outPrc
                                           ::testing::Values(Layout::ANY),                     // inLayout
                                           ::testing::Values(Layout::ANY),                     // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 3, 16, 16}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 2D Convolution / LargeSize ============= */

const auto conv2DParams_LargeSize1 =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{3, 3}}),              // kernels
                           ::testing::ValuesIn<SizeVector>({{2, 2}}),              // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                           ::testing::Values(64),                                  // numOutChannels
                           ::testing::Values(ngraph::op::PadType::VALID)           // padType
        );

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_LargeSize1, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_LargeSize1,                               //
                                           ::testing::Values(Precision::FP16),                    // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // outPrc
                                           ::testing::Values(Layout::ANY),                        // inLayout
                                           ::testing::Values(Layout::ANY),                        // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 128, 128}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        ConvolutionLayerTest::getTestCaseName);

const auto conv2DParams_LargeSize2 =
        ::testing::Combine(::testing::ValuesIn<SizeVector>({{3, 3}}),              // kernels
                           ::testing::ValuesIn<SizeVector>({{2, 2}}),              // strides
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padBegins
                           ::testing::ValuesIn<std::vector<ptrdiff_t>>({{0, 0}}),  // padEnds
                           ::testing::ValuesIn<SizeVector>({{1, 1}}),              // dilations
                           ::testing::Values(16),                                  // numOutChannels
                           ::testing::Values(ngraph::op::PadType::VALID)           // padType
        );

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_LargeSize2, KmbConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_LargeSize2,                               //
                                           ::testing::Values(Precision::FP16),                    // netPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // inPrc
                                           ::testing::Values(Precision::UNSPECIFIED),             // outPrc
                                           ::testing::Values(Layout::ANY),                        // inLayout
                                           ::testing::Values(Layout::ANY),                        // outLayout
                                           ::testing::ValuesIn<SizeVector>({{1, 16, 256, 256}}),  // inputShapes
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)),  //
                        ConvolutionLayerTest::getTestCaseName);

}  // namespace
