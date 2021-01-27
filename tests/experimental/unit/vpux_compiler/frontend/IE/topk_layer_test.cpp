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

typedef std::tuple<ngraph::element::Type, ngraph::Shape, ngraph::element::Type, int64_t> TopKTestParamsSet;

class IE_FrontEndTest_TopK : public testing::TestWithParam<TopKTestParamsSet> {};

TEST_P(IE_FrontEndTest_TopK, TopKLayer) {
    ngraph::element::Type dataType;
    ngraph::Shape dataShape;
    ngraph::element::Type indexElementType;
    int64_t axisConstant;

    std::tie(dataType, dataShape, indexElementType, axisConstant) = this->GetParam();

    std::shared_ptr<ngraph::Function> f;
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(dataType, dataShape);
        auto kConstant = ngraph::op::Constant::create(ngraph::element::Type_t::i64, {}, {axisConstant});
        auto topK = std::make_shared<ngraph::opset1::TopK>(data, kConstant, 1, "min", "value", indexElementType);
        topK->set_friendly_name("TopK");
        auto result1 = std::make_shared<ngraph::op::Result>(topK->output(0));
        auto result2 = std::make_shared<ngraph::op::Result>(topK->output(1));

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2}, ngraph::ParameterVector{data});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}
const std::vector<ngraph::element::Type> inputDataType{
        ngraph::element::f16,
        ngraph::element::f32,
};
const std::vector<ngraph::Shape> dataShape{{6, 12, 10, 24}};

const std::vector<ngraph::element::Type> indexElementType{
        ngraph::element::i32,
        ngraph::element::i64,
};

const std::vector<int64_t> kConstant{{1, 2, 3}};

const auto topKParams = ::testing::Combine(::testing::ValuesIn(inputDataType), ::testing::ValuesIn(dataShape),
                                           ::testing::ValuesIn(indexElementType), ::testing::ValuesIn(kConstant));
INSTANTIATE_TEST_CASE_P(IE_FrontEndTest_TopK_TestCase, IE_FrontEndTest_TopK, topKParams);
