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

typedef std::tuple<ngraph::Shape, int64_t, int64_t> SplitTestParamsSet;

class MLIR_IE_FrontEndTest_Split : public testing::TestWithParam<SplitTestParamsSet> {};

TEST_P(MLIR_IE_FrontEndTest_Split, SplitLayer) {

    ngraph::Shape dataShape;
    int64_t axis;
    int64_t numSplits;

    std::tie(dataShape, axis, numSplits) = this->GetParam();

    std::shared_ptr<ngraph::Function> f;
    {
        auto param1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, dataShape);
        auto axisConstant = ngraph::op::Constant::create(ngraph::element::Type_t::i64, {}, {axis});
        auto split = std::make_shared<ngraph::opset1::Split>(param1, axisConstant, numSplits);
        split->set_friendly_name("Split");
        ngraph::ResultVector results;
        results.reserve(static_cast<size_t>(numSplits));
        for(auto& i : split->outputs()) {
            results.emplace_back(std::make_shared<ngraph::op::Result>(i));
        }
        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{results}, ngraph::ParameterVector{param1});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}

const std::vector<ngraph::Shape> dataShape{{6, 12, 18, 24}};
const std::vector<int64_t> axes{-3, -2, -1, 0, 1, 2, 3};
const std::vector<int64_t> splits{1, 2, 3};

const auto splitParams = ::testing::Combine(::testing::ValuesIn(dataShape), ::testing::ValuesIn(axes), ::testing::ValuesIn(splits));
INSTANTIATE_TEST_CASE_P(MLIR_IE_FrontEndTest_Split_TestCase, MLIR_IE_FrontEndTest_Split, splitParams);
