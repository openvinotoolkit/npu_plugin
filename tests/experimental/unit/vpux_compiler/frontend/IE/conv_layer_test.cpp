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

std::shared_ptr<Node> makeConvolution(const ngraph::Output<Node>& in, const element::Type& type,
                                      const std::vector<size_t>& filterSize, const std::vector<size_t>& strides,
                                      const std::vector<ptrdiff_t>& padsBegin, const std::vector<ptrdiff_t>& padsEnd,
                                      const std::vector<size_t>& dilations, const op::PadType& autoPad,
                                      size_t numOutChannels) {
    auto shape = in.get_shape();

    // Make weights node
    std::vector<size_t> filterWeightsShape = {numOutChannels, shape[1]};
    filterWeightsShape.insert(filterWeightsShape.end(), filterSize.begin(), filterSize.end());

    size_t num_elements = 1;
    for (auto s : filterWeightsShape)
        num_elements *= s;
    std::shared_ptr<ngraph::Node> weightsNode = std::make_shared<ngraph::opset1::Constant>(
            type, filterWeightsShape, std::vector<ngraph::float16>(num_elements));

    // Make convolution
    auto conv = std::make_shared<opset1::Convolution>(in, weightsNode, strides, padsBegin, padsEnd, dilations, autoPad);
    return conv;
}

typedef std::tuple<std::vector<size_t>,     // Kernel size
                   std::vector<size_t>,     // Stride
                   std::vector<ptrdiff_t>,  // Pad begin
                   std::vector<ptrdiff_t>,  // Pad end
                   std::vector<size_t>,     // dilations
                   size_t,                  // Out channels
                   ngraph::op::PadType      // Pad type
                   >
        convSpecificParams;

typedef std::tuple<convSpecificParams,
                   InferenceEngine::Precision,  // Net precision
                   InferenceEngine::Precision,  // Input precision
                   InferenceEngine::Precision,  // Output precision
                   InferenceEngine::Layout,     // Input layout
                   InferenceEngine::Layout,     // Output layout
                   std::vector<size_t>          // Input shapes
                   >
        convLayerTestParamsSet;

class IE_FrontEndTest_Conv : public testing::TestWithParam<convLayerTestParamsSet> {};

TEST_P(IE_FrontEndTest_Conv, ConvLayer) {
    std::shared_ptr<ngraph::Function> f;
    {
        std::vector<size_t> kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        ngraph::op::PadType padType;
        ngraph::Shape inputSize;
        size_t convOutChannels;
        std::vector<size_t> filterShape;
        InferenceEngine::Precision netPrc, inPrc, outPrc; /* These are ignored */
        InferenceEngine::Layout inLayout, outLayout;      /* ignored */
        convSpecificParams convParams;

        std::tie(convParams, netPrc, inPrc, outPrc, inLayout, outLayout, inputSize) = this->GetParam();
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

        VPUX_THROW_UNLESS(netPrc == InferenceEngine::Precision::FP16, "Net precision should be fp16");
        auto ngPrc = ::ngraph::element::Type(::ngraph::element::Type_t::f16);

        auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, inputSize);
        filterShape.push_back(inputSize[inputSize.size() - 2]);
        filterShape.push_back(convOutChannels);
        std::copy(kernel.begin(), kernel.end(), std::back_inserter(filterShape));
        auto filter = std::make_shared<ngraph::opset1::Constant>(ngraph::element::f16, filterShape);

        const ngraph::Strides strides(stride);
        const ngraph::Strides dilations(dilation);
        const ngraph::CoordinateDiff pads_begin(padBegin.begin(), padBegin.end());
        const ngraph::CoordinateDiff pads_end(padEnd.begin(), padEnd.end());
        const ngraph::Shape kernel_shape(kernel);
        const auto auto_pad = padType;

        auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(makeConvolution(
                param1, ngPrc, kernel, stride, pads_begin, pads_end, dilations, auto_pad, convOutChannels));

        conv->set_friendly_name("Convolution");
        auto result = std::make_shared<ngraph::op::Result>(conv);

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

const InferenceEngine::Precision inPrc = InferenceEngine::Precision::U8;
const InferenceEngine::Precision outPrc = InferenceEngine::Precision::FP16;

const InferenceEngine::Layout inLayout = InferenceEngine::Layout::NHWC;
const InferenceEngine::Layout outLayout = InferenceEngine::Layout::NHWC;

/* ============= 2D Convolution ============= */
const std::vector<InferenceEngine::SizeVector> kernels = {{3, 3}, {3, 5}};
const std::vector<InferenceEngine::SizeVector> strides = {{1, 1}, {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}, {0, 3}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}, {0, 3}};
const std::vector<InferenceEngine::SizeVector> dilations = {{1, 1}, {3, 1}};
const std::vector<size_t> numOutCannels = {1, 5};
const std::vector<ngraph::op::PadType> padTypes = {ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID};

const auto conv2DParams_ExplicitPadding =
        ::testing::Combine(::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::ValuesIn(padBegins),
                           ::testing::ValuesIn(padEnds), ::testing::ValuesIn(dilations),
                           ::testing::ValuesIn(numOutCannels), ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides), ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})), ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutCannels), ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_ExplicitPadding, IE_FrontEndTest_Conv,
    ::testing::Combine(conv2DParams_ExplicitPadding,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(inPrc), ::testing::Values(outPrc),
        ::testing::Values(inLayout), ::testing::Values(outLayout),
        ::testing::Values(InferenceEngine::SizeVector({1, 3, 30, 30}))
        /*,
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)*/));
INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_AutoPadValid, IE_FrontEndTest_Conv,
    ::testing::Combine(conv2DParams_AutoPadValid,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(inPrc), ::testing::Values(outPrc),
        ::testing::Values(inLayout), ::testing::Values(outLayout),
        ::testing::Values(InferenceEngine::SizeVector({1, 3, 30, 30}))
        /*,
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)*/));

/* ============= 3D Convolution ============= */
const std::vector<InferenceEngine::SizeVector> kernels3d = {{3, 3, 3}, {3, 5, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0}, {0, 2, 0}};

const std::vector<InferenceEngine::SizeVector> strides3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<InferenceEngine::SizeVector> dilations3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<size_t> numOutChannels3d = {5};

const auto conv3DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d), ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(paddings3d), ::testing::ValuesIn(dilations3d), ::testing::ValuesIn(numOutChannels3d),
        ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto conv3DParams_AutoPadValid =
        ::testing::Combine(::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d),
                           ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
                           ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})), ::testing::ValuesIn(dilations3d),
                           ::testing::ValuesIn(numOutChannels3d), ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(smoke_Convolution3D_ExplicitPadding, IE_FrontEndTest_Conv,
    ::testing::Combine(conv3DParams_ExplicitPadding,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(inPrc), ::testing::Values(outPrc),
        ::testing::Values(inLayout), ::testing::Values(outLayout),
        ::testing::Values(InferenceEngine::SizeVector({1, 3, 10, 10, 10}))
        /*,
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)*/));

INSTANTIATE_TEST_CASE_P(smoke_Convolution3D_AutoPadValid, IE_FrontEndTest_Conv,
    ::testing::Combine(conv3DParams_AutoPadValid,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(inPrc), ::testing::Values(outPrc),
        ::testing::Values(inLayout), ::testing::Values(outLayout),
        ::testing::Values(InferenceEngine::SizeVector({1, 3, 10, 10, 10}))
        /*,
        ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)*/));
}  // namespace
