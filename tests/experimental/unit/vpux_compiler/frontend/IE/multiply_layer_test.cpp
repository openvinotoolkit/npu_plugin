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

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl));
}

//
// IE_AutoBroadcastType_NONE_OR_EXPLICIT
//  : f32, f16, i64, i32, i16, i8, u64, u16, u8
//

TEST(IE_FrontendTest, Multiply_AutoBroadcastType_NONE_OR_EXPLICIT_f32) {
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{256, 56});
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{256, 56});
    testImportNetwork(param1, param2);
}

TEST(IE_FrontendTest, Multiply_AutoBroadcastType_NONE_OR_EXPLICIT_f16) {
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{256, 56});
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{256, 56});
    testImportNetwork(param1, param2);
}

//
// IE_AutoBroadcastType_NUMPY (MultiDirectional)
//  : f32, f16
//

TEST(IE_FrontendTest, Multiply_AutoBroadcastType_NUMPY_DIRECTIONAL_f32) {
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{8, 1, 6, 1});
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{7, 1, 5});
    testImportNetwork(param1, param2, ngraph::op::AutoBroadcastType::NUMPY);
}

TEST(IE_FrontendTest, Multiply_AutoBroadcastType_NUMPY_DIRECTIONAL_f16) {
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{8, 1, 6, 1});
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{7, 1, 5});
    testImportNetwork(param1, param2, ngraph::op::AutoBroadcastType::NUMPY);
}

TEST(IE_FrontendTest, Multiply_AutoBroadcastType_NUMPY_UNDIRECTIONAL_f32) {
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{7, 1, 5});
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{8, 1, 6, 1});
    testImportNetwork(param1, param2, ngraph::op::AutoBroadcastType::NUMPY);
}

TEST(IE_FrontendTest, Multiply_AutoBroadcastType_NUMPY_UNDIRECTIONAL_f16) {
    auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{7, 1, 5});
    auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{8, 1, 6, 1});
    testImportNetwork(param1, param2, ngraph::op::AutoBroadcastType::NUMPY);
}
