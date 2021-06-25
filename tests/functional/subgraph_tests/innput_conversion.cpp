// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>

namespace {

class KmbQuantizedInputConversionTest : public LayerTestsUtils::KmbLayerTestsCommon,
                                     public testing::WithParamInterface<LayerTestsUtils::TargetDevice> {
    void SetUp() override {
        const InferenceEngine::SizeVector inputShape{1, 3, 352, 352};
        const InferenceEngine::SizeVector weightsShape{32, 3, 3, 3};

        const auto params = ngraph::builder::makeParams(ngraph::element::f16, {inputShape});
        const auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const std::vector<float> dataLow = {-10.0f};
        const std::vector<float> dataHigh = {10.0f};

        const auto inputConvFq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngraph::element::f16, 256,
                                                                    {}, dataLow, dataHigh, dataLow, dataHigh);


        const auto weightsU8 = ngraph::builder::makeConstant<uint8_t>(ngraph::element::u8, weightsShape, {}, true, 255, 1);
        const auto weightsFP16 = std::make_shared<ngraph::opset1::Convert>(weightsU8, ngraph::element::f16);
        const std::vector<float> weightsDataLow = {0.0f};
        const std::vector<float> weightsDataHigh = {183.0f};


        const auto weightsFq = ngraph::builder::makeFakeQuantize(weightsFP16, ngraph::element::f16, 256,
                                                                    {}, weightsDataLow, weightsDataHigh, weightsDataLow, weightsDataHigh);

        const ngraph::Strides strides = {1, 1};
        const ngraph::CoordinateDiff pads_begin = {0, 0};
        const ngraph::CoordinateDiff pads_end = {0, 0};
        const ngraph::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ngraph::opset2::Convolution>(inputConvFq, weightsFq,
                                                                        strides, pads_begin, pads_end, dilations);


        const std::vector<float> outputConvLow = {0.0f};
        const std::vector<float> outputConvHigh = {154.0f};

        const auto outputConvFq = ngraph::builder::makeFakeQuantize(conv, ngraph::element::f16, 256,
                                                                   {}, outputConvLow, outputConvHigh, outputConvLow, outputConvHigh);

        const ngraph::ResultVector results{
                std::make_shared<ngraph::opset1::Result>(outputConvFq)
        };
        function = std::make_shared<ngraph::Function>(results, params, "KmbQuantizedConvWithInputConversion");
        targetDevice = GetParam();
        threshold = 0.1f;
    }

    void SkipBeforeValidate() override {
        if (isCompilerMCM()) {
            throw LayerTestsUtils::KmbSkipTestException("Comparison fails");
        }
    }

};


TEST_P(KmbQuantizedInputConversionTest, CompareWithRefs_MCM) {
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke, KmbQuantizedInputConversionTest,
    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
);

}
