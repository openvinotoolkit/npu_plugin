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

#include "ngraph/ngraph.hpp"
#include <gtest/gtest.h>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>

#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include "ngraph_mcm_frontend/passes/fuse_scale_in_previous_weights_fq.hpp"
#include <legacy/ngraph_ops/power.hpp>

class ScaleFusingNgraphPass : public ::testing::Test {
public:

protected:
    void SetUp() override;
    std::shared_ptr<ngraph::Function> BuildFunction();
};

std::shared_ptr<ngraph::Function> ScaleFusingNgraphPass::BuildFunction() {
    auto paramNode = std::make_shared<ngraph::op::Parameter>(
            ngraph::element::Type_t::f32, ngraph::Shape(std::vector<size_t>{ {1, 8, 1, 1}}));
    paramNode->set_friendly_name("Parameter");

    auto inputLow = std::make_shared<ngraph::op::v0::Constant>(
            ngraph::element::f32,
            ngraph::Shape(std::vector<size_t>{1}),
            std::vector<float>{0.f});
    auto inputHigh = std::make_shared<ngraph::op::v0::Constant>(
            ngraph::element::f32,
            ngraph::Shape({1}),
            std::vector<float>{1.f});
    auto outputLow = std::make_shared<ngraph::op::v0::Constant>(
            ngraph::element::f32,
            ngraph::Shape({1}),
            std::vector<float>{0.f});
    auto outputHigh = std::make_shared<ngraph::op::v0::Constant>(
            ngraph::element::f32,
            ngraph::Shape({1}),
            std::vector<float>{1.f});

    auto inputFq = std::make_shared<ngraph::op::v0::FakeQuantize>(
            paramNode->output(0), inputLow, inputHigh, outputLow, outputHigh, 255);

    auto inputLowW = std::make_shared<ngraph::op::v0::Constant>(
            ngraph::element::f32,
            ngraph::Shape(std::vector<size_t>{1}),
            std::vector<float>{0.f});
    auto inputHighW = std::make_shared<ngraph::op::v0::Constant>(
            ngraph::element::f32,
            ngraph::Shape({1}),
            std::vector<float>{1.f});
    auto outputLowW = std::make_shared<ngraph::op::v0::Constant>(
            ngraph::element::f32,
            ngraph::Shape({1}),
            std::vector<float>{0.f});
    auto outputHighW = std::make_shared<ngraph::op::v0::Constant>(
            ngraph::element::f32,
            ngraph::Shape({1}),
            std::vector<float>{1.f});

    auto convFirstShape = ngraph::Shape{32, 8, 1, 1};
    auto convolutionWeights = std::make_shared<ngraph::op::v0::Constant>(
            ngraph::element::Type_t::f32, convFirstShape, std::vector<float>(32 * 8, 0.5));

    auto weightsFq = std::make_shared<ngraph::op::v0::FakeQuantize>(
            convolutionWeights, inputLow, inputHigh, outputLow, outputHigh, 255);

    std::vector<ptrdiff_t> padBegin{0, 0};
    std::vector<ptrdiff_t> padEnd{0, 0};
    auto convolutionNodeFirst = std::make_shared<ngraph::op::v1::Convolution>(
            inputFq->output(0), weightsFq->output(0), ngraph::Strides(std::vector<size_t>{ 1, 1 }),
            ngraph::CoordinateDiff(padBegin), ngraph::CoordinateDiff(padEnd), ngraph::Strides(std::vector<size_t>{ 1, 1 }));

    auto addFirstShape = ngraph::Shape{ 1, 32, 1, 1 };
    auto addFirstConstantNode = std::make_shared<ngraph::op::Constant>(
            ngraph::element::Type_t::f32, addFirstShape, std::vector<float>(32, 0.5));

    auto addNodeFirst =
            std::make_shared<ngraph::op::v1::Add>(convolutionNodeFirst->output(0), addFirstConstantNode->output(0));
    auto clamp = std::make_shared<ngraph::op::Clamp>(addNodeFirst->output(0), 0, 6);

    auto scaleConst = std::make_shared<ngraph::op::v0::Constant>(
            ngraph::element::f32,
            ngraph::Shape({1, 1, 1, 1}),
            std::vector<float>{0.5f});
    auto multiply = std::make_shared<ngraph::op::v1::Multiply>(clamp->output(0), scaleConst);
    auto result_full = std::make_shared<ngraph::op::Result>(multiply->output(0));
    return std::make_shared<ngraph::Function>(
            result_full, ngraph::ParameterVector{ paramNode }, "testSubEfficient");
}

void ScaleFusingNgraphPass::SetUp() {
}

TEST_F(ScaleFusingNgraphPass, canCompile) {
    auto nGraphFunction = BuildFunction();

    ngraph::pass::Manager passManager;

    passManager.register_pass<ngraph::pass::ConvertOpSet3ToOpSet2>();
    passManager.register_pass<ngraph::pass::ConvertOpSet2ToOpSet1>();
    passManager.register_pass<ngraph::pass::ConvertOpSet1ToLegacy>();
    passManager.register_pass<ngraph::pass::ConstantFolding>();

    passManager.register_pass<FuseScaleAfterClamp>();
    passManager.run_passes(nGraphFunction);

    for (const auto& op : nGraphFunction->get_ordered_ops()) {
        ASSERT_FALSE(op->get_type_info() == ngraph::op::PowerIE::type_info);
    }
}
