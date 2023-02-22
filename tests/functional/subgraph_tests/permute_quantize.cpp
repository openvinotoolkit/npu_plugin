// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"
#include "subgraph_tests/nce_tasks.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class PermuteQuantizeTest_VPU3720 :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<std::tuple<InferenceEngine::SizeVector>> {
    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NCHW);
        cnnNetwork.getOutputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
    }
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice;
        const auto inputShape = std::get<0>(GetParam());

        const auto params = ngraph::builder::makeParams(ngraph::element::f16, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const auto nceTask = NCETasksHelpers::buildNCETask(paramOuts.at(0), NCETasksHelpers::NCEOpType::GroupConv2d);
        const auto quantRange = std::array<float, 4>{0.f, 255.f, 0.f, 255.f};
        const auto quantOp = NCETasksHelpers::quantize(nceTask, quantRange);
        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(quantOp)};

        function = std::make_shared<ngraph::Function>(results, params, "PermuteQuantizeTest_VPU3720");

        threshold = 0.5f;

        configuration["PERFORMANCE_HINT"] = "LATENCY";
        configuration["VPUX_DPU_GROUPS"] = "2";
    }
};

TEST_P(PermuteQuantizeTest_VPU3720, CompareWithRefs_MLIR_VPU3720) {
    useCompilerMLIR();
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

const std::vector<InferenceEngine::SizeVector> inputShapes = {
        {1, 3, 224, 224},
        {1, 3, 128, 256},
        {1, 1, 64, 64},
};

INSTANTIATE_TEST_SUITE_P(conv2d_with_act, PermuteQuantizeTest_VPU3720,
                         ::testing::Combine(::testing::ValuesIn(inputShapes)));

}  // namespace
