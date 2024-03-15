// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <ov_models/builders.hpp>
#include <ov_models/utils/ov_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

using namespace ov::test;

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

using QuantizedConvTestParams = std::tuple<ov::element::Type,  // inPrc
                                           ov::element::Type,  // outPrc
                                           std::vector<float>  // fqRanges
                                           >;
class QuantizedConvSubGraphTestCommon :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<QuantizedConvTestParams> {
    void SetUp() override {
        std::vector<float> dataFQRanges;
        std::tie(inType, outType, dataFQRanges) = GetParam();
        rel_threshold = 0.1f;

        const ov::Shape inputShape{1, 3, 62, 62};
        const ov::Shape weightsShape{48, 3, 3, 3};

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        const size_t dataLevels = 256;
        const std::vector<float> dataInLow = {dataFQRanges.at(0)};
        const std::vector<float> dataInHigh = {dataFQRanges.at(1)};
        const std::vector<float> dataOutLow = {dataFQRanges.at(2)};
        const std::vector<float> dataOutHigh = {dataFQRanges.at(3)};
        const auto dataFq = ngraph::builder::makeFakeQuantize(params[0], ov::element::f32, dataLevels, {}, dataInLow,
                                                              dataInHigh, dataOutLow, dataOutHigh);

        const auto weightsU8 = ngraph::builder::makeConstant<uint8_t>(ov::element::u8, weightsShape, {}, true, 254, 0);

        const auto weightsFP32 = std::make_shared<ov::op::v0::Convert>(weightsU8, ov::element::f32);

        const size_t weightsLevels = 255;

        const auto weightsInLow = ngraph::builder::makeConstant<float>(ov::element::f32, {1}, {0.0f}, false);
        const auto weightsInHigh = ngraph::builder::makeConstant<float>(ov::element::f32, {1}, {254.0f}, false);

        std::vector<float> perChannelLow(weightsShape[0]);
        std::vector<float> perChannelHigh(weightsShape[0]);

        for (size_t i = 0; i < weightsShape[0]; ++i) {
            perChannelLow[i] = 0.0f;
            perChannelHigh[i] = 1.0f;
        }

        const auto weightsOutLow = ngraph::builder::makeConstant<float>(ov::element::f32, {weightsShape[0], 1, 1, 1},
                                                                        perChannelLow, false);
        const auto weightsOutHigh = ngraph::builder::makeConstant<float>(ov::element::f32, {weightsShape[0], 1, 1, 1},
                                                                         perChannelHigh, false);

        const auto weightsFq = std::make_shared<ov::op::v0::FakeQuantize>(weightsFP32, weightsInLow, weightsInHigh,
                                                                          weightsOutLow, weightsOutHigh, weightsLevels);

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        const auto conv =
                std::make_shared<ov::op::v1::Convolution>(dataFq, weightsFq, strides, pads_begin, pads_end, dilations);
        const std::vector<float> outDataLow = {0.0f};
        const std::vector<float> outDataHigh = {255.0f};
        const auto outFq = ngraph::builder::makeFakeQuantize(conv, ov::element::f32, dataLevels, {}, outDataLow,
                                                             outDataHigh, outDataLow, outDataHigh);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outFq)};
        function = std::make_shared<ov::Model>(results, params, "QuantizedConv");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<QuantizedConvTestParams> obj) {
        ov::element::Type ip;
        ov::element::Type op;
        std::vector<float> fqRanges;
        std::tie(ip, op, fqRanges) = obj.param;

        std::ostringstream result;
        result << "InputPrec=" << ip << "_";
        result << "OutputPrec=" << op << "_";
        result << "FQ={" << fqRanges.at(0) << ", " << fqRanges.at(1) << ", " << fqRanges.at(2) << ", " << fqRanges.at(3)
               << "}_";
        return result.str();
    }
};

class QuantizedConvSubGraphTest_NPU3700 : public QuantizedConvSubGraphTestCommon {};
class QuantizedConvSubGraphTest_NPU3720 : public QuantizedConvSubGraphTestCommon {};

TEST_P(QuantizedConvSubGraphTest_NPU3700, SW) {
    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3700);
}

TEST_P(QuantizedConvSubGraphTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3700);
}

TEST_P(QuantizedConvSubGraphTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3720);
}

std::vector<std::vector<float>> fqRanges = {
        {0.0f, 255.0f, 0.0f, 255.0f},
        {0.0f, 244.0f, 0.0f, 244.0f},
        {0.0f, 255.0f, -1.0f, 1.0f},
        {0.0f, 244.0f, -1.0f, 1.0f},
};

const std::vector<ov::element::Type> netPrecisions = {ov::element::u8, ov::element::f16};

const std::vector<ov::element::Type> netOutputPrecisions = {ov::element::u8, ov::element::f32};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(netPrecisions), ::testing::ValuesIn(netOutputPrecisions),
                                           ::testing::ValuesIn(fqRanges));

INSTANTIATE_TEST_SUITE_P(smoke_QuantizedConv, QuantizedConvSubGraphTest_NPU3700, basicCases,
                         QuantizedConvSubGraphTestCommon::getTestCaseName);

std::vector<std::vector<float>> fqRangesM = {{0.0f, 255.0f, 0.0f, 255.0f}};

const std::vector<ov::element::Type> netPrecisionsM = {ov::element::f16};

const std::vector<ov::element::Type> netOutputPrecisionsM = {ov::element::f16};

const auto basicCasesM = ::testing::Combine(::testing::ValuesIn(netPrecisionsM),
                                            ::testing::ValuesIn(netOutputPrecisionsM), ::testing::ValuesIn(fqRangesM));

INSTANTIATE_TEST_SUITE_P(smoke_QuantizedConv, QuantizedConvSubGraphTest_NPU3720, basicCasesM,
                         QuantizedConvSubGraphTestCommon::getTestCaseName);

}  // namespace
