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

typedef std::tuple<ngraph::element::Type, ngraph::Shape, int, float, ngraph::op::EpsMode> NormalizeTestParamsSet;

class IE_FrontEndTest_Normalize : public testing::TestWithParam<NormalizeTestParamsSet> {};

TEST_P(IE_FrontEndTest_Normalize, NormalizeLayer) {
    ngraph::element::Type dataType;
    ngraph::Shape dataShape;
    int axis;
    float eps;
    ngraph::op::EpsMode epsMod;

    std::tie(dataType, dataShape, axis, eps, epsMod) = this->GetParam();

    std::shared_ptr<ngraph::Function> f;
    {
        auto param1 = std::make_shared<ngraph::opset1::Parameter>(dataType, dataShape);
        auto axisConstant = ngraph::op::Constant::create(ngraph::element::Type_t::i64, {}, {axis});
        auto normalize = std::make_shared<ngraph::opset1::NormalizeL2>(param1, axisConstant, eps, epsMod);
        normalize->set_friendly_name("Normalize");
        auto result = std::make_shared<ngraph::op::Result>(normalize);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{param1});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}

const std::vector<ngraph::element::Type> inputDataType{ngraph::element::f16, ngraph::element::f32};
const std::vector<ngraph::Shape> dataShape{{6, 12, 10, 24}};
const std::vector<int> axis{0, 1, 2, 3};
const std::vector<float> eps{1e-8f};
const std::vector<ngraph::op::EpsMode> epsMode{ngraph::op::EpsMode::ADD, ngraph::op::EpsMode::MAX};

const auto normalizeParams =
        ::testing::Combine(::testing::ValuesIn(inputDataType), ::testing::ValuesIn(dataShape),
                           ::testing::ValuesIn(axis), ::testing::ValuesIn(eps), ::testing::ValuesIn(epsMode));
INSTANTIATE_TEST_CASE_P(IE_FrontEndTest_Normalize_TestCase, IE_FrontEndTest_Normalize, normalizeParams);
