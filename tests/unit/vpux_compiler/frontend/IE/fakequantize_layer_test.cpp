//
// Copyright 2020 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <memory>

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/init_node_info.hpp>

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/frontend/IE.hpp"

TEST(MLIR_IE_FrontEndTest, FakeQuantizeLayerTest) {
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

TEST(MLIR_IE_FrontEndTest, FakeQuantizeLayerTest_AutoBroadcastNone) {
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
