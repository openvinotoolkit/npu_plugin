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

typedef std::tuple<ngraph::element::Type, ngraph::Shape, bool, std::pair<int, int>> RegionYoloTestParamsSet;

class IE_FrontEndTest_RegionYolo : public testing::TestWithParam<RegionYoloTestParamsSet> {};

TEST_P(IE_FrontEndTest_RegionYolo, RegionYoloLayer) {
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
INSTANTIATE_TEST_CASE_P(IE_FrontEndTest_RegionYolo_TestCase, IE_FrontEndTest_RegionYolo, regioYoloParams);
