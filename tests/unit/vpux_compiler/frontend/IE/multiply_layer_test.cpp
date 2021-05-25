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

static void testImportNetwork(std::shared_ptr<ngraph::opset1::Parameter> param1,
                              std::shared_ptr<ngraph::opset1::Parameter> param2,
                              ngraph::op::AutoBroadcastType autobType = ngraph::op::AutoBroadcastType::NONE) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto autob = ngraph::op::AutoBroadcastSpec(autobType);
        auto multiply = std::make_shared<ngraph::opset1::Multiply>(param1, param2, autob);
        multiply->set_friendly_name("Multiply");
        auto result = std::make_shared<ngraph::op::Result>(multiply);
        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1, param2});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }

    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;
    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}

//
// IE_AutoBroadcastType_NONE_OR_EXPLICIT
//  : f32, f16, i64, i32, i16, i8, u64, u16, u8
//

TEST(MLIR_IE_FrontEndTest, Multiply_AutoBroadcastType_NONE_OR_EXPLICIT_f32) {
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{256, 56});
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{256, 56});
    testImportNetwork(param1, param2);
}

TEST(MLIR_IE_FrontEndTest, Multiply_AutoBroadcastType_NONE_OR_EXPLICIT_f16) {
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{256, 56});
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{256, 56});
    testImportNetwork(param1, param2);
}

//
// IE_AutoBroadcastType_NUMPY (MultiDirectional)
//  : f32, f16
//

TEST(MLIR_IE_FrontEndTest, Multiply_AutoBroadcastType_NUMPY_DIRECTIONAL_f32) {
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{8, 1, 6, 1});
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{7, 1, 5});
    testImportNetwork(param1, param2, ngraph::op::AutoBroadcastType::NUMPY);
}

TEST(MLIR_IE_FrontEndTest, Multiply_AutoBroadcastType_NUMPY_DIRECTIONAL_f16) {
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{8, 1, 6, 1});
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{7, 1, 5});
    testImportNetwork(param1, param2, ngraph::op::AutoBroadcastType::NUMPY);
}

TEST(MLIR_IE_FrontEndTest, Multiply_AutoBroadcastType_NUMPY_UNDIRECTIONAL_f32) {
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{7, 1, 5});
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{8, 1, 6, 1});
    testImportNetwork(param1, param2, ngraph::op::AutoBroadcastType::NUMPY);
}

TEST(MLIR_IE_FrontEndTest, Multiply_AutoBroadcastType_NUMPY_UNDIRECTIONAL_f16) {
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{7, 1, 5});
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{8, 1, 6, 1});
    testImportNetwork(param1, param2, ngraph::op::AutoBroadcastType::NUMPY);
}
