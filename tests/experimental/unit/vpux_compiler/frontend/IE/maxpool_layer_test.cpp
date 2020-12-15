//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <memory>

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/init_node_info.hpp>

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/frontend/IE.hpp"

// Test code copied and simplified from:
// kmb-plugin/tests/functional/shared_tests_instances/single_layer_tests/pooling.cpp
enum PoolingTypes { MAX, AVG };
typedef std::tuple<std::vector<size_t>,       // Input size
                   PoolingTypes,              // Pooling type, max or avg
                   std::vector<size_t>,       // Kernel size
                   std::vector<size_t>,       // Stride
                   std::vector<size_t>,       // Pad begin
                   std::vector<size_t>,       // Pad end
                   ngraph::op::RoundingType,  // Rounding type
                   ngraph::op::PadType,       // Pad type
                   bool                       // Exclude pad
                   >
        poolSpecificParams;

class IE_FrontEndTest : public testing::TestWithParam<poolSpecificParams> {};

TEST_P(IE_FrontEndTest, MaxPoolLayer) {
    std::shared_ptr<ngraph::Function> f;
    {
        PoolingTypes poolType;
        std::vector<size_t> kernel, stride;
        std::vector<size_t> padBegin, padEnd;
        ngraph::op::PadType padType;
        ngraph::op::RoundingType roundingType;
        ngraph::Shape inputSize;
        bool excludePad;

        std::tie(inputSize, poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) =
                this->GetParam();

        auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, inputSize);
        const ngraph::Strides strides(stride);
        const ngraph::Shape pads_begin(padBegin);
        const ngraph::Shape pads_end(padEnd);
        const ngraph::Shape kernel_shape(kernel);
        const auto rounding_mode = roundingType;
        const auto auto_pad = padType;
        auto maxpool = std::make_shared<ngraph::opset1::MaxPool>(param1, strides, pads_begin, pads_end, kernel_shape,
                                                                 rounding_mode, auto_pad);
        maxpool->set_friendly_name("Maxpool");
        auto result = std::make_shared<ngraph::op::Result>(maxpool);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl));
}

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};
const InferenceEngine::SizeVector inputSize = {1, 3, 64, 64};
const std::vector<InferenceEngine::SizeVector> kernels = {{3, 3}, {3, 5}};
const std::vector<InferenceEngine::SizeVector> strides = {{1, 1}, {1, 2}};
const std::vector<InferenceEngine::SizeVector> padBegins = {{0, 0}, {0, 2}};
const std::vector<InferenceEngine::SizeVector> padEnds = {{0, 0}, {0, 2}};
const std::vector<ngraph::op::RoundingType> roundingTypes = {ngraph::op::RoundingType::CEIL,
                                                             ngraph::op::RoundingType::FLOOR};

////* ========== Max Polling ========== */
/* +========== Test for 1d kernel ========== */
const auto maxPool_1d_Kernel_Param = ::testing::Combine(
        ::testing::Values(InferenceEngine::SizeVector{1, 3, 100}),                               /* input */
        ::testing::Values(PoolingTypes::MAX), ::testing::Values(InferenceEngine::SizeVector{1}), /* kernel */
        ::testing::Values(InferenceEngine::SizeVector{2}),                                       /* stride */
        ::testing::Values(InferenceEngine::SizeVector{0}), ::testing::Values(InferenceEngine::SizeVector{0}), /* pads */
        ::testing::Values(ngraph::op::RoundingType::FLOOR), ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_CASE_P(MaxPool_1d_Kernel, IE_FrontEndTest, maxPool_1d_Kernel_Param);
/* +========== Explicit Pad Floor Rounding ========== */

const auto maxPool_ExplicitPad_FloorRounding_Params = ::testing::Combine(
        ::testing::Values(inputSize), ::testing::Values(PoolingTypes::MAX), ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides), ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds),
        ::testing::Values(ngraph::op::RoundingType::FLOOR), ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_CASE_P(MaxPool_ExplicitPad_FloorRounding, IE_FrontEndTest, maxPool_ExplicitPad_FloorRounding_Params);

/* ========== Explicit Pad Ceil Rounding ========== */
const auto maxPool_ExplicitPad_CeilRounding_Params = ::testing::Combine(
        ::testing::Values(inputSize), ::testing::Values(PoolingTypes::MAX), ::testing::ValuesIn(kernels),
        // TODO: Non 1 strides fails in ngraph reference implementation with error "The end corner is out of bounds at
        // axis 3" thrown in the test body.
        ::testing::Values(InferenceEngine::SizeVector({1, 1})), ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds), ::testing::Values(ngraph::op::RoundingType::CEIL),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_CASE_P(MaxPool_ExplicitPad_CeilRounding, IE_FrontEndTest, maxPool_ExplicitPad_CeilRounding_Params);
////* ========== Max & Avg Polling Cases ========== */
/*    ========== Valid Pad Rounding Not Applicable ========== */
const auto allPools_ValidPad_Params = ::testing::Combine(
        ::testing::Values(inputSize), ::testing::Values(PoolingTypes::MAX /*, PoolingTypes::AVG */),
        ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
        ::testing::Values(InferenceEngine::SizeVector({0, 0})), ::testing::Values(InferenceEngine::SizeVector({0, 0})),
        ::testing::Values(ngraph::op::RoundingType::FLOOR),  // placeholder value - Rounding Type not applicable for
                                                             // Valid pad type
        ::testing::Values(ngraph::op::PadType::VALID),
        ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling
INSTANTIATE_TEST_CASE_P(MaxPool_ValidPad, IE_FrontEndTest, allPools_ValidPad_Params);
