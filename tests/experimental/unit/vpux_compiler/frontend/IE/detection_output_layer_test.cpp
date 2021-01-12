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

typedef std::tuple<ngraph::element::Type, ngraph::Shape, ngraph::Shape, ngraph::Shape, int>
        DetectionOutputTestParamsSet;

class IE_FrontEndTest_DetectionOutput : public testing::TestWithParam<DetectionOutputTestParamsSet> {};

TEST_P(IE_FrontEndTest_DetectionOutput, DetectionOutputLayer) {
    ngraph::element::Type dataType;
    ngraph::Shape boxlogitsShape;
    ngraph::Shape classPredsShape;
    ngraph::Shape proposalsShape;
    int keepTopK;

    std::tie(dataType, boxlogitsShape, classPredsShape, proposalsShape, keepTopK) = this->GetParam();

    ngraph::op::DetectionOutputAttrs attr;
    attr.background_label_id = 0;
    attr.code_type = "caffe.PriorBoxParameter.CENTER_SIZE";
    attr.input_height = 1;
    attr.input_width = 1;
    attr.keep_top_k = {keepTopK};
    attr.nms_threshold = 0.44999998807907104;
    attr.normalized = true;
    attr.num_classes = 21;
    attr.share_location = true;
    attr.top_k = 400;
    attr.variance_encoded_in_target = false;

    std::shared_ptr<ngraph::Function> f;
    {
        auto boxlogits = std::make_shared<ngraph::opset1::Parameter>(dataType, boxlogitsShape);
        auto classPreds = std::make_shared<ngraph::opset1::Parameter>(dataType, classPredsShape);
        auto proposals = std::make_shared<ngraph::opset1::Parameter>(dataType, proposalsShape);

        auto detectionOutput =
                std::make_shared<ngraph::opset1::DetectionOutput>(boxlogits, classPreds, proposals, attr);
        detectionOutput->set_friendly_name("DetectionOutput");
        auto result = std::make_shared<ngraph::op::Result>(detectionOutput);

        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                               ngraph::ParameterVector{boxlogits, classPreds, proposals});
        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}

const std::vector<ngraph::element::Type> inputDataType{ngraph::element::f16, ngraph::element::f32};
const std::vector<ngraph::Shape> boxLogitsShapes{{1, 98256}};
const std::vector<ngraph::Shape> classPredsShapes{{1, 515844}};
const std::vector<ngraph::Shape> proposalsShapes{{1, 2, 98256}};
const std::vector<int> keepTopK{20, 40, 80, 200};

const auto DetectionOutputParams = ::testing::Combine(
        ::testing::ValuesIn(inputDataType), ::testing::ValuesIn(boxLogitsShapes), ::testing::ValuesIn(classPredsShapes),
        ::testing::ValuesIn(proposalsShapes), ::testing::ValuesIn(keepTopK));
INSTANTIATE_TEST_CASE_P(IE_FrontEndTest_DetectionOutput_TestCase, IE_FrontEndTest_DetectionOutput,
                        DetectionOutputParams);
