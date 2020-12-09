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

TEST(IE_FrontEndTest, GatherLayer) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{6, 12, 10, 24});
        std::vector<int64_t> indicesVector{2, 2, 4, 5, 2, 2, 4, 5, 2, 2, 4, 5};
        auto indices =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{4, 3}, indicesVector);
        auto axisConstant = ngraph::op::Constant::create(ngraph::element::Type_t::i64, {}, {1});
        auto gather = std::make_shared<ngraph::opset1::Gather>(param1, indices, axisConstant);
        gather->set_friendly_name("Gather");
        auto result = std::make_shared<ngraph::op::Result>(gather);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl));
}
