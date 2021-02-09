// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>

#include <transformations/init_node_info.hpp>
#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/frontend/IE.hpp"

// This test code is copied from KMB functional test of convolution. It was modified to test more comprehensively.
// As this is only for front-end testing, it tests un-supported configuration as well, such as 3d conv or dilated conv.

namespace {

using namespace ngraph;

std::shared_ptr<Node> makeConvolutionBackpropData(
        const ngraph::Output<Node>& in, const std::vector<size_t>& outputShape, const element::Type& type,
        const std::vector<size_t>& filterSize, const std::vector<size_t>& strides,
        const std::vector<ptrdiff_t>& padsBegin, const std::vector<ptrdiff_t>& padsEnd,
        const std::vector<size_t>& dilations, const op::PadType& autoPad, size_t numOutChannels) {
    auto shape = in.get_shape();

    // Make weights node
    std::vector<size_t> filterWeightsShape = {shape[1], numOutChannels};
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());

    size_t num_elements = 1;
    for (auto s : filterWeightsShape)
        num_elements *= s;

    std::shared_ptr<ngraph::Node> weightsNode = std::make_shared<ngraph::opset1::Constant>(
            type, filterWeightsShape, std::vector<ngraph::float16>(num_elements));

    if (outputShape.size() > 0) {
        // Make outputshape node
        std::shared_ptr<ngraph::Node> outputShapeNode = std::make_shared<ngraph::opset1::Constant>(
                ngraph::element::i64, ngraph::Shape({outputShape.size()}), outputShape);

        return std::make_shared<opset1::ConvolutionBackpropData>(in, weightsNode, outputShapeNode, strides, padsBegin,
                                                                 padsEnd, dilations, autoPad);
    } else {
        return std::make_shared<opset1::ConvolutionBackpropData>(in, weightsNode, strides, padsBegin, padsEnd,
                                                                 dilations, autoPad);
    }
}

typedef std::tuple<InferenceEngine::SizeVector,  // Kernel size
                   InferenceEngine::SizeVector,  // Strides
                   std::vector<ptrdiff_t>,       // Pad begin
                   std::vector<ptrdiff_t>,       // Pad end
                   InferenceEngine::SizeVector,  // Dilation
                   size_t,                       // Num out channels
                   ngraph::op::PadType           // Padding type
                   >
        convBackpropDataSpecificParams;

typedef std::tuple<convBackpropDataSpecificParams,
                   InferenceEngine::Precision,   // Net precision
                   InferenceEngine::Precision,   // Input precision
                   InferenceEngine::Precision,   // Output precision
                   InferenceEngine::Layout,      // Input layout
                   InferenceEngine::Layout,      // Output layout
                   InferenceEngine::SizeVector,  // Input shapes
                   InferenceEngine::SizeVector   // Output shapes
                   >
        convBackpropDataLayerTestParamsSet;

class MLIR_IE_FrontEndTest_Decovolution : public testing::TestWithParam<convBackpropDataLayerTestParamsSet> {};

TEST_P(MLIR_IE_FrontEndTest_Decovolution, DecovolutionLayer) {
    std::shared_ptr<ngraph::Function> f;
    {
        convBackpropDataSpecificParams convBackpropDataParams;
        InferenceEngine::Precision netPrecision;
        InferenceEngine::Precision inPrc, outPrc;
        InferenceEngine::Layout inLayout, outLayout;
        ngraph::Shape inputSize;

        ngraph::op::PadType padType;
        InferenceEngine::SizeVector kernel, stride, dilation, outputShapeValues;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels;
        std::tie(convBackpropDataParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputSize,
                 outputShapeValues) = this->GetParam();

        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convBackpropDataParams;

        VPUX_THROW_UNLESS(netPrecision == InferenceEngine::Precision::FP16, "Net precision should be fp16");
        auto ngPrc = ::ngraph::element::Type(::ngraph::element::Type_t::f16);
        auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, inputSize);
        auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i64,
                                                                  ngraph::PartialShape{outputShapeValues}.get_shape());

        const ngraph::Strides strides(stride);
        const ngraph::Strides dilations(dilation);
        const ngraph::CoordinateDiff pads_begin(padBegin.begin(), padBegin.end());
        const ngraph::CoordinateDiff pads_end(padEnd.begin(), padEnd.end());
        const ngraph::Shape kernel_shape(kernel);

        auto convBackpropData = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(
                makeConvolutionBackpropData(param1, outputShapeValues, ngPrc, kernel, stride, pads_begin, pads_end,
                                            dilations, padType, convOutChannels));

        convBackpropData->set_friendly_name("Deconvolution");
        auto result = std::make_shared<ngraph::op::Result>(convBackpropData);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1, param2});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const InferenceEngine::Precision inPrc = InferenceEngine::Precision::U8;
const InferenceEngine::Precision outPrc = InferenceEngine::Precision::FP16;

const InferenceEngine::Layout inLayout = InferenceEngine::Layout::NHWC;
const InferenceEngine::Layout outLayout = InferenceEngine::Layout::NHWC;

const std::vector<size_t> numOutChannels = {1, 5, 16};

/* ============= 2D Deconvolution (w/o output shape) ============= */
const std::vector<std::vector<size_t>> inputShapes2D_1 = {{1, 3, 30, 30}, {1, 16, 10, 10}, {1, 32, 10, 10}};
const std::vector<std::vector<size_t>> outputShapes2D_1 = {{}};  // output shape not specified
const std::vector<std::vector<size_t>> kernels2D_1 = {{1, 1}, {3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> strides2D_1 = {{1, 1}, {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D_1 = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D_1 = {{0, 0}, {1, 1}};
const std::vector<std::vector<size_t>> dilations2D_1 = {{1, 1}, {2, 2}};

const auto conv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels2D_1), ::testing::ValuesIn(strides2D_1), ::testing::ValuesIn(padBegins2D_1),
        ::testing::ValuesIn(padEnds2D_1), ::testing::ValuesIn(dilations2D_1), ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto conv2DParams_AutoPadValid =
        ::testing::Combine(::testing::ValuesIn(kernels2D_1), ::testing::ValuesIn(strides2D_1),
                           ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                           ::testing::Values(std::vector<ptrdiff_t>({0, 0})), ::testing::ValuesIn(dilations2D_1),
                           ::testing::ValuesIn(numOutChannels), ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(DeconvolutionOp2D_ExplicitPadding, MLIR_IE_FrontEndTest_Decovolution,
                        ::testing::Combine(conv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes2D_1),
                                           ::testing::ValuesIn(outputShapes2D_1)));

INSTANTIATE_TEST_CASE_P(DeconvolutionOp2D_AutoPadValid, MLIR_IE_FrontEndTest_Decovolution,
                        ::testing::Combine(conv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes2D_1),
                                           ::testing::ValuesIn(outputShapes2D_1)));

/* ============= 2D Deconvolution (w/ output shape) ============= */
const std::vector<std::vector<size_t>> inputShapes2D_2 = {{1, 3, 30, 30}};
const std::vector<std::vector<size_t>> outputShapes2D_2 = {{80, 80}};
const std::vector<std::vector<size_t>> kernels2D_2 = {{3, 3}};
const std::vector<std::vector<size_t>> strides2D_2 = {{2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D_2 = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D_2 = {{0, 0}};
const std::vector<std::vector<size_t>> dilations2D_2 = {{1, 1}};

const auto conv2DParams_AutoPadSameLower =
        ::testing::Combine(::testing::ValuesIn(kernels2D_2), ::testing::ValuesIn(strides2D_2),
                           ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                           ::testing::Values(std::vector<ptrdiff_t>({0, 0})), ::testing::ValuesIn(dilations2D_2),
                           ::testing::ValuesIn(numOutChannels), ::testing::Values(ngraph::op::PadType::EXPLICIT));

INSTANTIATE_TEST_CASE_P(DeconvolutionOp2D_AutoPadValid_OutputShape, MLIR_IE_FrontEndTest_Decovolution,
                        ::testing::Combine(conv2DParams_AutoPadSameLower, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes2D_2),
                                           ::testing::ValuesIn(outputShapes2D_2)));

/* ============= 3D Deconvolution (w/o output) ============= */
const std::vector<std::vector<size_t>> inputShapes3D_1 = {{1, 3, 10, 10, 10}, {1, 16, 5, 5, 5}, {1, 32, 5, 5, 5}};
const std::vector<std::vector<size_t>> outputShapes3D_1 = {{}};
const std::vector<std::vector<size_t>> kernels3D_1 = {{1, 1, 1}, {3, 3, 3}};
const std::vector<std::vector<size_t>> strides3D_1 = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D_1 = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D_1 = {{0, 0, 0}, {1, 1, 1}};
const std::vector<std::vector<size_t>> dilations3D_1 = {{1, 1, 1}, {2, 2, 2}};

const auto conv3DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels3D_1), ::testing::ValuesIn(strides3D_1), ::testing::ValuesIn(padBegins3D_1),
        ::testing::ValuesIn(padEnds3D_1), ::testing::ValuesIn(dilations3D_1), ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto conv3DParams_AutoPadValid =
        ::testing::Combine(::testing::ValuesIn(kernels3D_1), ::testing::ValuesIn(strides3D_1),
                           ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
                           ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), ::testing::ValuesIn(dilations3D_1),
                           ::testing::ValuesIn(numOutChannels), ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(DeconvolutionOp3D_ExplicitPadding, MLIR_IE_FrontEndTest_Decovolution,
                        ::testing::Combine(conv3DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes3D_1),
                                           ::testing::ValuesIn(outputShapes3D_1)));

INSTANTIATE_TEST_CASE_P(DeconvolutionOp3D_AutoPadValid, MLIR_IE_FrontEndTest_Decovolution,
                        ::testing::Combine(conv3DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes3D_1),
                                           ::testing::ValuesIn(outputShapes3D_1)));

/* ============= 3D Deconvolution (w/ output) ============= */
const std::vector<std::vector<size_t>> inputShapes3D_2 = {{1, 3, 5, 30, 30}};
const std::vector<std::vector<size_t>> outputShapes3D_2 = {{5, 80, 80}};
const std::vector<std::vector<size_t>> kernels3D_2 = {{5, 3, 3}};
const std::vector<std::vector<size_t>> strides3D_2 = {{5, 2, 2}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D_2 = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D_2 = {{0, 0, 0}};
const std::vector<std::vector<size_t>> dilations3D_2 = {{1, 1, 1}};

const auto conv3DParams_ExplicitPadding_OutputShape = ::testing::Combine(
        ::testing::ValuesIn(kernels3D_2), ::testing::ValuesIn(strides3D_2), ::testing::ValuesIn(padBegins3D_2),
        ::testing::ValuesIn(padEnds3D_2), ::testing::ValuesIn(dilations3D_2), ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT));

INSTANTIATE_TEST_CASE_P(DeconvolutionOp3D_ExplicitPadding_OutputShape, MLIR_IE_FrontEndTest_Decovolution,
                        ::testing::Combine(conv3DParams_ExplicitPadding_OutputShape, ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::Values(InferenceEngine::Layout::ANY),
                                           ::testing::ValuesIn(inputShapes3D_2),
                                           ::testing::ValuesIn(outputShapes3D_2)));

}  // namespace
