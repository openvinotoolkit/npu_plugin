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

TEST(IE_FrontendPowerTest, ImportNetwork) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{256, 56});
        auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{256, 56});
        auto autob = ngraph::op::AutoBroadcastSpec();
        auto pow = std::make_shared<ngraph::opset1::Power>(param1, param2, autob);
        pow->set_friendly_name("Power");
        auto result = std::make_shared<ngraph::op::Result>(pow);
        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1, param2});

        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;
    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    vpux::IE::FrontEnd frontEnd(&ctx);
    EXPECT_NO_THROW(frontEnd.importNetwork(nGraphImpl));
}

TEST(IE_FrontendPowerTest, AutoBroadcastType_NUMPY) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{8, 1, 6, 1});
        auto param2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{7, 1, 5});
        auto pow = std::make_shared<ngraph::opset1::Power>(param1, param2);
        pow->set_friendly_name("Power");
        auto result = std::make_shared<ngraph::op::Result>(pow);
        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1, param2});

        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;
    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    vpux::IE::FrontEnd frontEnd(&ctx);
    EXPECT_NO_THROW(frontEnd.importNetwork(nGraphImpl));
}
