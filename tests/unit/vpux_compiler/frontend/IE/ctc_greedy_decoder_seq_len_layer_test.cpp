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
#include <ngraph/opsets/opset6.hpp>
#include <transformations/init_node_info.hpp>

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/frontend/IE.hpp"

TEST(MLIR_IE_FrontEndTest, CTCGreedyDecoderSeqLenLayer) {
    std::shared_ptr<ngraph::Function> f;
    {
        const auto N = 20;
        const auto T = 40;
        const auto C = 60;

        const auto inputShape = ngraph::Shape{N, T, C};
        auto param = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, inputShape);

        std::vector<int32_t> sequenceLength(N);
        std::fill(begin(sequenceLength), end(sequenceLength), T);
        auto seqLenNode =
                std::make_shared<ngraph::opset6::Constant>(ngraph::element::i32, ngraph::Shape{N}, sequenceLength);

        std::vector<int32_t> blankIndex{0};
        auto blankIndexNode =
                std::make_shared<ngraph::opset6::Constant>(ngraph::element::i32, ngraph::Shape{1}, blankIndex);

        auto mergeRepeated = true;
        auto ctcGreedyDecoderSeqLen = std::make_shared<ngraph::opset6::CTCGreedyDecoderSeqLen>(
                param, seqLenNode, blankIndexNode, mergeRepeated);
        auto result = std::make_shared<ngraph::op::Result>(ctcGreedyDecoderSeqLen);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}
