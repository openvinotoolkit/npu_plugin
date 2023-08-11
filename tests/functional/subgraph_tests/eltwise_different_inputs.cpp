// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class VPUXEltwiseAddQuantizedSubGraphTest_VPU3720 :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<std::tuple<LayerTestsUtils::TargetDevice>> {
    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getInputsInfo().at("input2")->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    }
    void SetUp() override {
        const InferenceEngine::SizeVector inputShape{1, 16, 56, 56};
        const InferenceEngine::SizeVector weightsShape{1, 16, 56, 56};

        auto params = ngraph::builder::makeParams(ngraph::element::f16, {inputShape, weightsShape});
        params[0]->set_friendly_name("input1");
        params[1]->set_friendly_name("input2");
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const size_t dataLevels = 256;
        const auto dataFq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngraph::element::f16, dataLevels, {}, {0.0},
                                                              {12.583984375}, {0.0}, {12.583984375});

        const size_t weightsLevels = 256;
        const auto weightsFq = ngraph::builder::makeFakeQuantize(paramOuts[1], ngraph::element::f16, weightsLevels, {},
                                                                 {0.0}, {2.583984375}, {0.0}, {2.583984375});

        const auto addOp = std::make_shared<ngraph::opset1::Add>(dataFq, weightsFq);

        const size_t outLevels = 256;
        const auto outputFq = ngraph::builder::makeFakeQuantize(addOp, ngraph::element::f16, outLevels, {}, {0.0},
                                                                {13.583984375}, {0.0}, {13.583984375});

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(outputFq)};
        function = std::make_shared<ngraph::Function>(results, params, "EltwiseAddQuantized");

        targetDevice = std::get<0>(GetParam());
        threshold = 0.1f;
    }
};

TEST_P(VPUXEltwiseAddQuantizedSubGraphTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke_EltwiseAddQuantized, VPUXEltwiseAddQuantizedSubGraphTest_VPU3720,
                        ::testing::Combine(::testing::Values(LayerTestsUtils::testPlatformTargetDevice)));

}  // namespace
