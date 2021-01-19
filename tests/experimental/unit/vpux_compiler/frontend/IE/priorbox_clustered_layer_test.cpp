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

TEST(IE_FrontEndTest, PriorBoxClusteredLayer) {
    std::shared_ptr<ngraph::Function> f;
    {
        // Test vector copied from sample in
        //   https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/detection/PriorBoxClustered_1.md
        std::shared_ptr<ngraph::Node> outputSize = std::make_shared<ngraph::opset1::Constant>(
                ngraph::element::i64, std::vector<size_t>{2}, std::vector<ngraph::float16>{10, 19});
        auto imageSize = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i64, ngraph::Shape{2});

        ngraph::op::PriorBoxClusteredAttrs attrs;
        attrs.widths = std::vector<float>{86.0f, 13.0f, 57.0f, 39.0f, 68.0f, 34.0f, 142.0f, 50.0f, 23.0f};
        attrs.heights = std::vector<float>{44.0f, 10.0f, 30.0f, 19.0f, 94.0f, 32.0f, 61.0f, 53.0f, 17.0f};
        attrs.clip = false;
        attrs.step_widths = 16.0f;
        attrs.step_heights = 16.0f;
        attrs.offset = 0.5f;
        attrs.variances = std::vector<float>{0.1f, 0.1f, 0.2f, 0.2f};
        auto priorBoxClustered = std::make_shared<ngraph::opset1::PriorBoxClustered>(outputSize, imageSize, attrs);
        priorBoxClustered->set_friendly_name("PriorBoxClustered");
        auto result = std::make_shared<ngraph::op::Result>(priorBoxClustered);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{imageSize});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}
