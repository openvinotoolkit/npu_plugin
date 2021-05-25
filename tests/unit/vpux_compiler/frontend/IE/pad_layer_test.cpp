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

typedef std::tuple<ngraph::element::Type, ngraph::Shape, std::vector<int64_t>, std::vector<int64_t>,
                   std::pair<ngraph::op::PadMode, bool>, float>
        PadTestParamsSet;

class MLIR_IE_FrontEndTest_Pad : public testing::TestWithParam<PadTestParamsSet> {};

TEST_P(MLIR_IE_FrontEndTest_Pad, PadLayer) {
    ngraph::element::Type dataType;
    ngraph::Shape dataShape;
    std::vector<int64_t> pads_begin;
    std::vector<int64_t> pads_end;
    std::pair<ngraph::op::PadMode, bool> pad_mode;
    float pad_value;
    std::tie(dataType, dataShape, pads_begin, pads_end, pad_mode, pad_value) = this->GetParam();

    std::shared_ptr<ngraph::Function> f;
    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(dataType, dataShape);
        auto padsBegin = std::make_shared<ngraph::opset6::Constant>(ngraph::element::i64,
                                                                    ngraph::Shape{pads_begin.size()}, pads_begin);
        auto padsEnd = std::make_shared<ngraph::opset6::Constant>(ngraph::element::i64, ngraph::Shape{pads_end.size()},
                                                                  pads_end);
        auto padValue = ngraph::op::Constant::create(dataType, {}, {pad_value});

        std::cout << "provide pad value:" << (pad_mode.second ? "ON" : "OFF") << std::endl;
        auto pad = pad_mode.second
                           ? std::make_shared<ngraph::opset6::Pad>(data, padsBegin, padsEnd, padValue, pad_mode.first)
                           : std::make_shared<ngraph::opset6::Pad>(data, padsBegin, padsEnd, pad_mode.first);
        pad->set_friendly_name("Pad");
        auto result = std::make_shared<ngraph::op::Result>(pad->output(0));

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{data});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}

const std::vector<ngraph::element::Type> inputDataType{
        ngraph::element::i8,
        ngraph::element::f16,
        ngraph::element::f32,
};

const std::vector<ngraph::Shape> dataShape{{4, 3}};
const std::vector<std::vector<int64_t>> pads_begin{{0, 1}};
const std::vector<std::vector<int64_t>> pads_end{{2, 3}};
const std::vector<std::pair<ngraph::op::PadMode, bool>> pad_mode{
        {ngraph::op::PadMode::EDGE, false},     {ngraph::op::PadMode::REFLECT, false},
        {ngraph::op::PadMode::CONSTANT, false}, {ngraph::op::PadMode::SYMMETRIC, false},
        {ngraph::op::PadMode::CONSTANT, true},
};
const std::vector<float> pad_value{0, 1, 2};
const auto padParams =
        ::testing::Combine(::testing::ValuesIn(inputDataType), ::testing::ValuesIn(dataShape),
                           ::testing::ValuesIn(pads_begin), ::testing::ValuesIn(pads_end), ::testing::ValuesIn(pad_mode),
                           ::testing::ValuesIn(pad_value));
INSTANTIATE_TEST_CASE_P(MLIR_IE_FrontEndTest_Pad_TestCase, MLIR_IE_FrontEndTest_Pad, padParams);
