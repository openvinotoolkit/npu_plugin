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

TEST(IE_FrontEndTest, FakeQuantizeLayerTest) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 64, 56, 56});
        auto inputLow = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 64, 1, 1});
        auto inputHigh = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 64, 1, 1});
        auto outputLow = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1});
        auto outputHigh = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1});
        size_t levels = 4;
        auto fakeQuan = std::make_shared<ngraph::opset1::FakeQuantize>(data, inputLow, inputHigh, outputLow, outputHigh,
                                                                       levels);
        fakeQuan->set_friendly_name("FakeQuantize");
        auto result = std::make_shared<ngraph::op::Result>(fakeQuan);
        f = std::make_shared<ngraph::Function>(
                ngraph::ResultVector{result},
                ngraph::ParameterVector{data, inputLow, inputHigh, outputLow, outputHigh});

        ngraph::pass::InitNodeInfo().run_on_function(f);
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;
    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}

TEST(IE_FrontEndTest, FakeQuantizeLayerTest_AutoBroadcastNone) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 64, 56, 56});
        auto inputLow = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 64, 56, 56});
        auto inputHigh =
                std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 64, 56, 56});
        auto outputLow =
                std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 64, 56, 56});
        auto outputHigh =
                std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 64, 56, 56});
        size_t levels = 4;
        auto autob = ngraph::op::AutoBroadcastSpec(ngraph::op::AutoBroadcastType::NONE);
        auto fakeQuan = std::make_shared<ngraph::opset1::FakeQuantize>(data, inputLow, inputHigh, outputLow, outputHigh,
                                                                       levels, autob);
        fakeQuan->set_friendly_name("FakeQuantize");
        auto result = std::make_shared<ngraph::op::Result>(fakeQuan);
        f = std::make_shared<ngraph::Function>(
                ngraph::ResultVector{result},
                ngraph::ParameterVector{data, inputLow, inputHigh, outputLow, outputHigh});

        ngraph::pass::InitNodeInfo().run_on_function(f);
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;
    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}
