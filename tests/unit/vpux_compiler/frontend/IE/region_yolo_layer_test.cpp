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

typedef std::tuple<ngraph::element::Type, ngraph::Shape, bool, std::pair<int, int>> RegionYoloTestParamsSet;

class MLIR_IE_FrontEndTest_RegionYolo : public testing::TestWithParam<RegionYoloTestParamsSet> {};

TEST_P(MLIR_IE_FrontEndTest_RegionYolo, RegionYoloLayer) {
    ngraph::element::Type dataType;
    ngraph::Shape dataShape;
    bool doSoftmax;
    std::pair<int, int> beginAndEndAxis;

    std::tie(dataType, dataShape, doSoftmax, beginAndEndAxis) = this->GetParam();

    std::shared_ptr<ngraph::Function> f;
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(dataType, dataShape);
        auto regionYolo =
                std::make_shared<ngraph::opset1::RegionYolo>(data, 4, 80, 0, doSoftmax, std::vector<int64_t>{0, 1, 2},
                                                             beginAndEndAxis.first, beginAndEndAxis.second);
        regionYolo->set_friendly_name("RegionYolo");
        auto result = std::make_shared<ngraph::op::Result>(regionYolo);

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
        ngraph::element::f16,
        ngraph::element::f32,
};
const std::vector<ngraph::Shape> dataShape{{1, 255, 26, 26}, {1, 125, 13, 13}};
const std::vector<bool> doSoftmax{true, false};
const std::vector<std::pair<int, int>> beginAndEndAxis{{1, 1}, {1, 2}, {1, 3}};

const auto regioYoloParams = ::testing::Combine(::testing::ValuesIn(inputDataType), ::testing::ValuesIn(dataShape),
                                                ::testing::ValuesIn(doSoftmax), ::testing::ValuesIn(beginAndEndAxis));
INSTANTIATE_TEST_CASE_P(MLIR_IE_FrontEndTest_RegionYolo_TestCase, MLIR_IE_FrontEndTest_RegionYolo, regioYoloParams);
