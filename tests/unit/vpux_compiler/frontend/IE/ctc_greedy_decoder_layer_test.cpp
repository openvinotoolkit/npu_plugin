//
// Copyright 2021 Intel Corporation.
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

TEST(MLIR_IE_FrontEndTest, CTCGreedyDecoderLayer) {
    std::shared_ptr<ngraph::Function> f;
    {
        const auto T = 88;
        const auto N = 1;
        const auto C = 71;

        const auto inputShape = ngraph::Shape{T, N, C};
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, inputShape);

        std::vector<float> seqLenData(T * N);
        std::fill(begin(seqLenData), end(seqLenData), 1.0f);
        auto seqLenNode =
                std::make_shared<ngraph::opset1::Constant>(ngraph::element::f32, ngraph::Shape{T, N}, seqLenData);

        auto mergeRepeated = true;
        auto ctcGreedyDecoder = std::make_shared<ngraph::opset1::CTCGreedyDecoder>(param, seqLenNode, mergeRepeated);
        auto result = std::make_shared<ngraph::op::Result>(ctcGreedyDecoder);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}

