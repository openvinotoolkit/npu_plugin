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

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/init_node_info.hpp>

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/frontend/IE.hpp"

typedef std::tuple<ngraph::element::Type,
                   ngraph::Shape,        // data shape
                   std::vector<size_t>,  // target spatial shape
                   std::string,          // iterpolate attr
                   ngraph::AxisSet>
        InterpolateTestParamsSet;

class IE_FrontEndTest_Interpolate : public testing::TestWithParam<InterpolateTestParamsSet> {};

TEST_P(IE_FrontEndTest_Interpolate, InterpolateLayer) {
    ngraph::element::Type elementType;
    ngraph::Shape inputShape;
    std::vector<size_t> targetSpatial;
    std::string mode;
    ngraph::AxisSet axes;

    std::tie(elementType, inputShape, targetSpatial, mode, axes) = this->GetParam();

    ngraph::op::InterpolateAttrs attr;
    attr.axes = axes;
    attr.mode = mode;
    attr.align_corners = false;
    attr.antialias = false;
    attr.pads_begin = {0};
    attr.pads_end = {0};

    std::shared_ptr<ngraph::Function> f;
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(elementType, inputShape);
        auto outputShapeDesc = std::make_shared<ngraph::opset1::Constant>(
                ngraph::element::i64, ngraph::Shape{targetSpatial.size()}, targetSpatial);
        auto interpolate = std::make_shared<ngraph::opset1::Interpolate>(data, outputShapeDesc, attr);
        interpolate->set_friendly_name("Interpolate");
        auto result = std::make_shared<ngraph::op::Result>(interpolate);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{data});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl));
}

const std::vector<ngraph::element::Type> netPrecisions = {
        ngraph::element::f16,
        ngraph::element::f32,
};

const std::vector<ngraph::Shape> inShapes{{1, 1, 23, 23}, {10, 3, 230, 230}};

const std::vector<std::vector<size_t>> targetShapes2D{{46, 46}, {12, 12}};

const std::vector<std::vector<size_t>> targetShapes4D{{1, 1, 46, 46}, {1, 1, 12, 12}};

const std::vector<std::string> modes{"linear", "cubic", "nearest", "area"};

const std::vector<ngraph::AxisSet> defaultAxes2D = {{2, 3}, {3, 2}};
const std::vector<ngraph::AxisSet> defaultAxes4D = {{0, 1, 2, 3}, {3, 2, 1, 0}};

const auto interpParams2D = ::testing::Combine(::testing::ValuesIn(netPrecisions), ::testing::ValuesIn(inShapes),
                                               ::testing::ValuesIn(targetShapes2D), ::testing::ValuesIn(modes),
                                               ::testing::ValuesIn(defaultAxes2D));
INSTANTIATE_TEST_CASE_P(IE_FrontEndTest_Interpolate_2DTarget, IE_FrontEndTest_Interpolate, interpParams2D);

const auto interpParams4D = ::testing::Combine(::testing::ValuesIn(netPrecisions), ::testing::ValuesIn(inShapes),
                                               ::testing::ValuesIn(targetShapes4D), ::testing::ValuesIn(modes),
                                               ::testing::ValuesIn(defaultAxes4D));
INSTANTIATE_TEST_CASE_P(IE_FrontEndTest_Interpolate_4DTarget, IE_FrontEndTest_Interpolate, interpParams4D);
