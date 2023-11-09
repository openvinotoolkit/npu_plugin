//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpu_ov1_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class VPUXMatMulWithTransposeTest_VPU3720 :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
        public testing::WithParamInterface<std::vector<int64_t>> {
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice();
        const auto transposeOrder = GetParam();
        const size_t inputColumns = 64;
        const size_t outputColumns = 76;
        const InferenceEngine::SizeVector lhsInputShape = {1, 4, 8, inputColumns};
        const InferenceEngine::SizeVector rhsInputShape = {1, 8, 4, inputColumns};

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {lhsInputShape, rhsInputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const auto lhsGeMM = buildMatMul(paramOuts.at(0), ngraph::Shape{1, 4, inputColumns, outputColumns});
        const auto lhsGeMMTranspose = buildTranspose(lhsGeMM->output(0), std::vector<int64_t>{0, 2, 1, 3});
        const auto rhsGeMM = buildMatMul(paramOuts.at(1), ngraph::Shape{1, 8, inputColumns, outputColumns});
        const auto add = buildAdd(lhsGeMMTranspose->output(0), rhsGeMM->output(0));
        const auto transpose = buildTranspose(add->output(0), transposeOrder);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset7::Result>(transpose)};

        function = std::make_shared<ngraph::Function>(results, params, "VPUXMatMulWithTransposeTest");

        threshold = 0.5f;
    }

    std::shared_ptr<ov::Node> buildMatMul(const ov::Output<ov::Node>& param, const ngraph::Shape& weightsShape) {
        const InferenceEngine::SizeVector inputShape = param.get_shape();
        const auto weightsSize =
                std::accumulate(weightsShape.cbegin(), weightsShape.cend(), 1, std::multiplies<size_t>());
        std::vector<float> values(weightsSize, 1.f);

        const auto weights = ngraph::opset8::Constant::create(ngraph::element::f32, weightsShape, values);
        return std::make_shared<ngraph::opset7::MatMul>(param, weights, false, false);
    }

    std::shared_ptr<ov::Node> buildTranspose(const ov::Output<ov::Node>& param, const std::vector<int64_t>& dimsOrder) {
        auto order = ngraph::opset8::Constant::create(ngraph::element::i64, {dimsOrder.size()}, dimsOrder);
        return std::make_shared<ngraph::opset7::Transpose>(param, order);
    }

    std::shared_ptr<ov::Node> buildAdd(const ov::Output<ov::Node>& lhs, const ov::Output<ov::Node>& rhs) {
        return std::make_shared<ngraph::opset7::Add>(lhs, rhs);
    }
};

TEST_P(VPUXMatMulWithTransposeTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXMatMulWithTransposeTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

const std::vector<std::vector<int64_t>> transposes = {
        {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_MatMulWithTranspose, VPUXMatMulWithTransposeTest_VPU3720,
                         ::testing::ValuesIn(transposes));

}  // namespace
