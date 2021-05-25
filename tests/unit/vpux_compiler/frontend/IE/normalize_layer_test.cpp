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

typedef std::tuple<ngraph::element::Type, ngraph::Shape, int, float, ngraph::op::EpsMode> NormalizeTestParamsSet;

class MLIR_IE_FrontEndTest_Normalize : public testing::TestWithParam<NormalizeTestParamsSet> {};

TEST_P(MLIR_IE_FrontEndTest_Normalize, NormalizeLayer) {
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
INSTANTIATE_TEST_CASE_P(MLIR_IE_FrontEndTest_Normalize_TestCase, MLIR_IE_FrontEndTest_Normalize, normalizeParams);
