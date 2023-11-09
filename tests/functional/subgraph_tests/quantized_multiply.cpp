//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpu_ov1_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

// MLIR detects pattern quant.dcast -> op -> quant.qcast and converts it into single quantized Op
//
//      [input 1]
//          |
//     (dequantize)
//          |
//      (Multiply) --- (dequantize) -- [input 2]
//          |
//       [output]
//          |
//      (quantize)
//

using QuantizedMulTestParams = std::tuple<InferenceEngine::Precision,  // inPrc
                                          InferenceEngine::Precision,  // outPrc
                                          std::vector<float>,          // fqRanges0
                                          std::vector<float>,          // fqRanges1
                                          LayerTestsUtils::TargetDevice>;

class VPUXQuantizedMulSubGraphTest_VPU3700 :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
        public testing::WithParamInterface<QuantizedMulTestParams> {
    void SetUp() override {
        std::vector<float> data0FQRanges, data1FQRanges;
        std::tie(inPrc, outPrc, data0FQRanges, data1FQRanges, targetDevice) = GetParam();
        threshold = 0.1f;

        const InferenceEngine::SizeVector shape{1, 16, 32, 32};

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {shape, shape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const auto makeFQdata = [](const ngraph::Output<ngraph::Node>& input, const std::vector<float>& dataRanges) {
            const size_t dataLevels = 256;
            const std::vector<float> dataInLow = {dataRanges.at(0)};
            const std::vector<float> dataInHigh = {dataRanges.at(1)};
            const std::vector<float> dataOutLow = {dataRanges.at(2)};
            const std::vector<float> dataOutHigh = {dataRanges.at(3)};
            return ngraph::builder::makeFakeQuantize(input, ngraph::element::f32, dataLevels, {}, dataInLow, dataInHigh,
                                                     dataOutLow, dataOutHigh);
        };

        const auto data0Fq = makeFQdata(paramOuts[0], data0FQRanges);
        const auto data1Fq = makeFQdata(paramOuts[1], data1FQRanges);

        const auto mul = std::make_shared<ngraph::op::v1::Multiply>(data0Fq->output(0), data1Fq->output(0));

        const std::vector<float> outputRange = {0.0f, 255.0f, 0.0f, 255.0f};
        const auto outFq = makeFQdata(mul, outputRange);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(outFq)};
        function = std::make_shared<ngraph::Function>(results, params, "VPUXQuantizedMul");
    }

public:
    static std::string getTestCaseName(
            testing::TestParamInfo<std::tuple<std::vector<float>, std::vector<float>, LayerTestsUtils::TargetDevice>>
                    obj) {
        std::vector<float> fqRanges0, fqRanges1;
        std::string targetDevice;
        std::tie(fqRanges0, fqRanges1, targetDevice) = obj.param;

        std::ostringstream result;
        result << "FQ0={" << fqRanges0.at(0) << ", " << fqRanges0.at(1) << ", " << fqRanges0.at(2) << ", "
               << fqRanges0.at(3) << "}_"
               << "FQ1={" << fqRanges1.at(0) << ", " << fqRanges1.at(1) << ", " << fqRanges1.at(2) << ", "
               << fqRanges1.at(3) << "}_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }
};

TEST_P(VPUXQuantizedMulSubGraphTest_VPU3700, SW) {
    setPlatformVPU3700();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXQuantizedMulSubGraphTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

std::vector<std::vector<float>> fqRanges = {
        {0.0f, 255.0f, 0.0f, 255.0f},
        {0.0f, 255.0f, -1.0f, 1.0f},
        {0.0f, 244.0f, -1.0f, 1.0f},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::U8,
                                                               InferenceEngine::Precision::FP16};

const auto basicCases =
        ::testing::Combine(::testing::ValuesIn(netPrecisions), ::testing::Values(InferenceEngine::Precision::FP32),
                           ::testing::ValuesIn(fqRanges), ::testing::ValuesIn(fqRanges),
                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice()));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_smoke_QuantizedMul, VPUXQuantizedMulSubGraphTest_VPU3700, basicCases);

}  // namespace
