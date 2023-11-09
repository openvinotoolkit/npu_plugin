//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpu_ov1_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class VPUXAddWithTransposeTest_VPU3720 :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
        public testing::WithParamInterface<std::vector<int64_t>> {
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice();
        const auto transposeOrder = GetParam();
        const InferenceEngine::SizeVector lhsInputShape = {1, 8, 4, 16};
        const InferenceEngine::SizeVector rhsInputShape = {1, 8, 4, 16};
        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {lhsInputShape, rhsInputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
        const auto lhsTranspose = buildTranspose(paramOuts.at(0), transposeOrder);
        const auto rhsTranspose = buildTranspose(paramOuts.at(1), transposeOrder);
        const auto add = buildAdd(lhsTranspose->output(0), rhsTranspose->output(0));

        const ngraph::ResultVector results{std::make_shared<ngraph::opset7::Result>(add)};

        function = std::make_shared<ngraph::Function>(results, params, "VPUXTransposeEltwise");

        threshold = 0.5f;
    }

    std::shared_ptr<ov::Node> buildTranspose(const ov::Output<ov::Node>& param, const std::vector<int64_t>& dimsOrder) {
        auto order = ngraph::opset8::Constant::create(ngraph::element::i64, {dimsOrder.size()}, dimsOrder);
        return std::make_shared<ngraph::opset7::Transpose>(param, order);
    }

    std::shared_ptr<ov::Node> buildAdd(const ov::Output<ov::Node>& lhs, const ov::Output<ov::Node>& rhs) {
        return std::make_shared<ngraph::opset7::Add>(lhs, rhs);
    }
};

TEST_P(VPUXAddWithTransposeTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXAddWithTransposeTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

const std::vector<std::vector<int64_t>> transposes = {
        {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
};

INSTANTIATE_TEST_SUITE_P(smoke_transpose_add, VPUXAddWithTransposeTest_VPU3720, ::testing::ValuesIn(transposes));

}  // namespace
