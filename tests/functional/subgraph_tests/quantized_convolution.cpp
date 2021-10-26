// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

// MLIR detects pattern quant.dcast -> op -> quant.qcast and converts it into single quantized Op
//
//       [input]
//          |
//     (dequantize)
//          |
//        (conv) --- (dequantize) -- [filter]
//          |
//       [output]
//          |
//      (quantize)
//

class KmbQuantizedConvSubGraphTest :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<std::tuple<std::vector<float>, LayerTestsUtils::TargetDevice>> {
    void SetUp() override {
        const InferenceEngine::SizeVector inputShape{1, 3, 62, 62};
        const InferenceEngine::SizeVector weightsShape{48, 3, 3, 3};

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const size_t dataLevels = 256;
        const auto dataFQRanges = std::get<0>(GetParam());
        const std::vector<float> dataInLow   = {dataFQRanges.at(0)};
        const std::vector<float> dataInHigh  = {dataFQRanges.at(1)};
        const std::vector<float> dataOutLow  = {dataFQRanges.at(2)};
        const std::vector<float> dataOutHigh = {dataFQRanges.at(3)};
        const auto dataFq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngraph::element::f32, dataLevels, {},
                                                              dataInLow, dataInHigh, dataOutLow, dataOutHigh);

        const auto weightsU8 =
                ngraph::builder::makeConstant<uint8_t>(ngraph::element::u8, weightsShape, {}, true, 254, 0);

        const auto weightsFP32 = std::make_shared<ngraph::opset2::Convert>(weightsU8, ngraph::element::f32);

        const size_t weightsLevels = 255;

        const auto weightsInLow = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1}, {0.0f}, false);
        const auto weightsInHigh = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1}, {254.0f}, false);

        std::vector<float> perChannelLow(weightsShape[0]);
        std::vector<float> perChannelHigh(weightsShape[0]);

        for (size_t i = 0; i < weightsShape[0]; ++i) {
            perChannelLow[i] = 0.0f;
            perChannelHigh[i] = 1.0f;
        }

        const auto weightsOutLow = ngraph::builder::makeConstant<float>(
                ngraph::element::f32, {weightsShape[0], 1, 1, 1}, perChannelLow, false);
        const auto weightsOutHigh = ngraph::builder::makeConstant<float>(
                ngraph::element::f32, {weightsShape[0], 1, 1, 1}, perChannelHigh, false);

        const auto weightsFq = std::make_shared<ngraph::opset2::FakeQuantize>(
                weightsFP32, weightsInLow, weightsInHigh, weightsOutLow, weightsOutHigh, weightsLevels);

        const ngraph::Strides strides = {1, 1};
        const ngraph::CoordinateDiff pads_begin = {0, 0};
        const ngraph::CoordinateDiff pads_end = {0, 0};
        const ngraph::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ngraph::opset2::Convolution>(dataFq, weightsFq, strides, pads_begin,
                                                                        pads_end, dilations);
        const std::vector<float> outDataLow = {0.0f};
        const std::vector<float> outDataHigh = {255.0f};
        const auto outFq = ngraph::builder::makeFakeQuantize(conv, ngraph::element::f32, dataLevels, {}, outDataLow, outDataHigh, outDataLow, outDataHigh);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(outFq)};
        function = std::make_shared<ngraph::Function>(results, params, "KmbQuantizedConv");

        targetDevice = std::get<1>(GetParam());
        threshold = 0.1f;
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<std::vector<float>, LayerTestsUtils::TargetDevice>> obj) {
        std::vector<float> fqRanges;
        std::string targetDevice;
        std::tie(fqRanges, targetDevice) = obj.param;

        std::ostringstream result;
        result << "FQ={" << fqRanges.at(0) << ", " << fqRanges.at(1) << ", " << fqRanges.at(2) << ", " << fqRanges.at(3) << "}_";
        result << "targetDevice=" << targetDevice;
        return result.str();
}
};

TEST_P(KmbQuantizedConvSubGraphTest, CompareWithRefs_MLIR_SW) {
    useCompilerMLIR();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(KmbQuantizedConvSubGraphTest, CompareWithRefs_MLIR_HW) {
    useCompilerMLIR();
    setReferenceHardwareModeMLIR();
    Run();
}

std::vector<std::vector<float>> fqRanges = {
        {0.0f, 255.0f, 0.0f, 255.0f},
        {0.0f, 255.0f, -1.0f, 1.0f},
};

INSTANTIATE_TEST_CASE_P(smoke, KmbQuantizedConvSubGraphTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(fqRanges),
                                ::testing::Values(LayerTestsUtils::testPlatformTargetDevice)
                                ),
                        KmbQuantizedConvSubGraphTest::getTestCaseName);
}  // namespace
