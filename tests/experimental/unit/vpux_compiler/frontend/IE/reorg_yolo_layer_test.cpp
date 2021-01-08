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
#include <ngraph/opsets/opset2.hpp>
#include <transformations/init_node_info.hpp>

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/frontend/IE.hpp"

typedef std::tuple<ngraph::element::Type, ngraph::Shape, int> ReorgYoloTestParamsSet;

class IE_FrontEndTest_ReorgYolo : public testing::TestWithParam<ReorgYoloTestParamsSet> {};

TEST_P(IE_FrontEndTest_ReorgYolo, ReorgYoloLayer) {
    ngraph::element::Type dataType;
    ngraph::Shape dataShape;
    int strides;

    std::tie(dataType, dataShape, strides) = this->GetParam();

    std::shared_ptr<ngraph::Function> f;
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(dataType, dataShape);
        auto reorgYolo = std::make_shared<ngraph::opset2::ReorgYolo>(data, strides);
        reorgYolo->set_friendly_name("ReorgYolo");
        auto result = std::make_shared<ngraph::op::Result>(reorgYolo);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{data});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}

const std::vector<ngraph::element::Type> inputDataType{ngraph::element::u32, ngraph::element::u64,
                                                       ngraph::element::i32, ngraph::element::i64,
                                                       ngraph::element::f16, ngraph::element::f32};
const std::vector<ngraph::Shape> dataShape{{1, 64, 16, 16}, {1, 125, 8, 64}};
const std::vector<int> strides{2, 4, 8};

const auto reorgYoloParams = ::testing::Combine(::testing::ValuesIn(inputDataType), ::testing::ValuesIn(dataShape),
                                                ::testing::ValuesIn(strides));
INSTANTIATE_TEST_CASE_P(IE_FrontEndTest_ReorgYolo_TestCase, IE_FrontEndTest_ReorgYolo, reorgYoloParams);
