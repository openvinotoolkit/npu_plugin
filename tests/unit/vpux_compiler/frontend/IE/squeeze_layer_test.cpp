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

TEST(MLIR_IE_FrontEndTest, SqueezeLayerTest_ImportNetwork) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 1, 2});
        // std::vector<int64_t> axesVec;
        auto axes = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{0});
        auto squeeze = std::make_shared<ngraph::opset1::Squeeze>(input, axes);
        squeeze->set_friendly_name("Squeeze");
        auto result = std::make_shared<ngraph::op::Result>(squeeze);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;
    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}

TEST(MLIR_IE_FrontEndTest, SqueezeLayerTest_Axes) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 1, 2});
        std::vector<int64_t> axesVec{0, 2};
        auto axes = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{2}, axesVec);
        auto squeeze = std::make_shared<ngraph::opset1::Squeeze>(input, axes);
        squeeze->set_friendly_name("Squeeze");
        auto result = std::make_shared<ngraph::op::Result>(squeeze);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;
    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}

TEST(MLIR_IE_FrontEndTest, SqueezeLayerTest_ZeroTensor) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1});
        std::vector<int64_t> axesVec{0};
        auto axes = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{1}, axesVec);
        auto squeeze = std::make_shared<ngraph::opset1::Squeeze>(input, axes);
        squeeze->set_friendly_name("Squeeze");
        auto result = std::make_shared<ngraph::op::Result>(squeeze);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;
    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}
