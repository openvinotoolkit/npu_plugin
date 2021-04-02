//
// Copyright 2021 Intel Corporation.
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
