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

TEST(IE_FrontEndTest, PriorBoxLayer) {
    std::shared_ptr<ngraph::Function> f;
    {
        // Test vector copied from sample in
        //   https://github.com/openvinotoolkit/openvino/blob/master/docs/ops/detection/PriorBox_1.md
        std::shared_ptr<ngraph::Node> outputSize = std::make_shared<ngraph::opset1::Constant>(
                ngraph::element::i64, std::vector<size_t>{2}, std::vector<ngraph::float16>{24, 42});
        auto imageSize = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i64, ngraph::Shape{2});

        ngraph::op::PriorBoxAttrs attrs;
        attrs.min_size = std::vector<float>{16.0f};
        attrs.max_size = std::vector<float>{38.46f};
        attrs.aspect_ratio = std::vector<float>{2.0f};
        attrs.clip = false;
        attrs.flip = true;
        attrs.step = 16.0f;
        attrs.offset = 0.5f;
        attrs.variance = std::vector<float>{0.1f, 0.1f, 0.2f, 0.2f};
        auto priorBox = std::make_shared<ngraph::opset1::PriorBox>(outputSize, imageSize, attrs);
        priorBox->set_friendly_name("PriorBox");
        auto result = std::make_shared<ngraph::op::Result>(priorBox);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{imageSize});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}
