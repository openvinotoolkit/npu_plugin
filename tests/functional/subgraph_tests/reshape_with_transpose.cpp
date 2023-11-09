//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpu_ov1_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class VPUXReshapeWithTransposeTest_VPU3720 :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
        public testing::WithParamInterface<std::vector<int64_t>> {
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice();
        const auto transposeOrder = GetParam();
        const size_t inputColumns = 64;
        const size_t outputColumns = 76;
        const InferenceEngine::SizeVector lhsInputShape = {1, 768, 14, 14};
        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {lhsInputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        const auto add = buildAdd(paramOuts.at(0));

        std::vector<size_t> newShape;
        newShape.push_back(lhsInputShape[0]);
        newShape.push_back(lhsInputShape[1]);
        newShape.push_back(lhsInputShape[2] * lhsInputShape[3]);
        const auto reshape = buildReshape(add, newShape);

        const auto lhsTranspose = buildTranspose(reshape, transposeOrder);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset7::Result>(lhsTranspose)};

        function = std::make_shared<ngraph::Function>(results, params, "VPUXReshapeTranspose");

        threshold = 0.5f;
    }

    std::shared_ptr<ov::Node> buildTranspose(const ov::Output<ov::Node>& param, const std::vector<int64_t>& dimsOrder) {
        auto order = ngraph::opset8::Constant::create(ngraph::element::i64, {dimsOrder.size()}, dimsOrder);
        return std::make_shared<ngraph::opset7::Transpose>(param, order);
    }

    std::shared_ptr<ov::Node> buildReshape(const ov::Output<ov::Node>& param, const std::vector<size_t>& newShape) {
        auto constNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                    ngraph::Shape{newShape.size()}, newShape);
        const auto reshape = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(
                std::make_shared<ngraph::opset1::Reshape>(param, constNode, false));
        return reshape;
    }

    std::shared_ptr<ov::Node> buildAdd(const ov::Output<ov::Node>& lhs) {
        const auto inShape = lhs.get_shape();
        const auto constShape = ngraph::Shape{1, inShape.at(1), 1, 1};
        std::vector<float> values(inShape.at(1), 1.f);
        const auto biasConst = ngraph::opset8::Constant::create(ngraph::element::f32, constShape, values);
        return std::make_shared<ngraph::opset7::Add>(lhs, biasConst);
    }
};

TEST_P(VPUXReshapeWithTransposeTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXReshapeWithTransposeTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

const std::vector<std::vector<int64_t>> transposes = {
        {0, 2, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_transpose_add, VPUXReshapeWithTransposeTest_VPU3720, ::testing::ValuesIn(transposes));

}  // namespace
