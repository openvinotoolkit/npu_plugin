// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace LayerTestsDefinitions {

// This test aims for:
//   - Check that Dequantize layer is propagated through the Clamp
//   - Check that Convolution's ppe has right value of clamp_high and clamp_low attributes,
//     not just min/max value of int type
//
//       [input]
//          |
//         FQ
//          |
//        (conv) --- FQ -- [filter]
//          |
//          FQ
//          |
//        Clamp
//          |
//       [output]
//

using outFQAndClampRangesType = std::vector<std::pair<float, float>>;

using QuantizedConvClampTestParams = std::tuple<InferenceEngine::Precision,  // inPrc
                                                InferenceEngine::Precision,  // outPrc
                                                outFQAndClampRangesType, LayerTestsUtils::TargetDevice>;

class VPUXQuantizedConvClampSubGraphTest :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<QuantizedConvClampTestParams> {
    void GenerateInputs() override {
        inputs.clear();
        const auto& inputsInfo = executableNetwork.GetInputsInfo();
        const auto& functionParams = function->get_parameters();
        for (size_t i = 0; i < functionParams.size(); ++i) {
            const auto& param = functionParams[i];
            const auto infoIt = inputsInfo.find(param->get_friendly_name());
            GTEST_ASSERT_NE(infoIt, inputsInfo.cend());
            InferenceEngine::InputInfo::CPtr info = infoIt->second;
            const uint32_t range = 100;
            const int32_t start_from = -50;
            InferenceEngine::Blob::Ptr blob =
                    FuncTestUtils::createAndFillBlob(info->getTensorDesc(), range, start_from);
            inputs.push_back(blob);
        }
    }

    void SetUp() override {
        outFQAndClampRangesType outFQAndClampRanges;
        std::tie(inPrc, outPrc, outFQAndClampRanges, targetDevice) = GetParam();
        threshold = 0.5f;
        inLayout = InferenceEngine::Layout::NHWC;

        const InferenceEngine::SizeVector inputShape{1, 16, 20, 20};
        const InferenceEngine::SizeVector weightsShape{32, 16, 1, 1};

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        // Create FQ for input
        const size_t dataLevels = 256;
        const std::vector<float> dataInLow = {-127};
        const std::vector<float> dataInHigh = {128};
        const std::vector<float> dataOutLow = {-127};
        const std::vector<float> dataOutHigh = {128};
        const auto dataFq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngraph::element::f32, dataLevels, {},
                                                              dataInLow, dataInHigh, dataOutLow, dataOutHigh);

        // Create FQ for weights
        const auto weightsU8 = ngraph::builder::makeConstant<uint8_t>(ngraph::element::u8, weightsShape, {}, true,
                                                                      /*upTo=*/1, /*startFrom=*/1);

        const auto weightsFP32 = std::make_shared<ngraph::opset2::Convert>(weightsU8, ngraph::element::f32);

        const size_t weightsLevels = 255;
        const auto weightsInLow = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1}, {-127.0f}, false);
        const auto weightsInHigh = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1}, {127.0f}, false);

        std::vector<float> perChannelLow(weightsShape[0]);
        std::vector<float> perChannelHigh(weightsShape[0]);

        for (size_t i = 0; i < weightsShape[0]; ++i) {
            perChannelLow[i] = -127.0f;
            perChannelHigh[i] = 127.0f;
        }

        const auto weightsOutLow = ngraph::builder::makeConstant<float>(
                ngraph::element::f32, {weightsShape[0], 1, 1, 1}, perChannelLow, false);
        const auto weightsOutHigh = ngraph::builder::makeConstant<float>(
                ngraph::element::f32, {weightsShape[0], 1, 1, 1}, perChannelHigh, false);

        const auto weightsFq = std::make_shared<ngraph::opset2::FakeQuantize>(
                weightsFP32, weightsInLow, weightsInHigh, weightsOutLow, weightsOutHigh, weightsLevels);

        // Create Convolution
        const ngraph::Strides strides = {1, 1};
        const ngraph::CoordinateDiff pads_begin = {0, 0};
        const ngraph::CoordinateDiff pads_end = {0, 0};
        const ngraph::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ngraph::opset2::Convolution>(dataFq, weightsFq, strides, pads_begin,
                                                                        pads_end, dilations);

        // Create out FQ
        auto outFQRanges = outFQAndClampRanges[0];
        const std::vector<float> outDataLow = {outFQRanges.first};
        const std::vector<float> outDataHigh = {outFQRanges.second};
        const auto outFq = ngraph::builder::makeFakeQuantize(conv, ngraph::element::f32, dataLevels, {}, outDataLow,
                                                             outDataHigh, outDataLow, outDataHigh);

        // Create Clamp
        const InferenceEngine::SizeVector convOutShape{1, 32, 20, 20};
        auto clampRanges = outFQAndClampRanges[1];
        std::vector<float> constantsValue{clampRanges.first, clampRanges.second};
        auto clamp = ngraph::builder::makeActivation(outFq, ngraph::element::f16, ngraph::helpers::Clamp, convOutShape,
                                                     constantsValue);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(clamp)};
        function = std::make_shared<ngraph::Function>(results, params, "VPUXQuantizedConvClamp");
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<QuantizedConvClampTestParams> obj) {
        InferenceEngine::Precision ip;
        InferenceEngine::Precision op;
        outFQAndClampRangesType outFQAndClampRanges;
        std::string targetDevice;
        std::tie(ip, op, outFQAndClampRanges, targetDevice) = obj.param;

        auto outFQRanges = outFQAndClampRanges[0];
        auto clampRanges = outFQAndClampRanges[1];

        std::ostringstream result;
        result << "InputPrec=" << ip.name() << "_";
        result << "OutputPrec=" << op.name() << "_";
        result << "outFQ={" << outFQRanges.first << ", " << outFQRanges.second << ", " << outFQRanges.first << ", "
               << outFQRanges.second << "}_";
        result << "clamp={" << clampRanges.first << ", " << clampRanges.second << "}_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }
};

class VPUXQuantizedConvClampSubGraphTest_VPU3720 : public VPUXQuantizedConvClampSubGraphTest {};

TEST_P(VPUXQuantizedConvClampSubGraphTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

std::vector<outFQAndClampRangesType> outFQAndClampRanges = {
        // FQ range > clamp range
        {{0.f, 10.f}, {0.f, 5.f}},
        {{-20.15748f, 20.0f}, {-5.f, 5.f}},
        {{-20.f, .0f}, {-5.f, 0.f}},
        // FQ range < clamp range
        {{-20.15748f, 20.0f}, {-25.f, 25.f}},
        // clamp range == FQ range
        {{-20.15748f, 20.0f}, {-20.15748f, 20.f}},
        // clamp range != FQ range
        {{-20.f, 0.0f}, {-10.f, 10.f}},
};

const std::vector<InferenceEngine::Precision> inPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<InferenceEngine::Precision> outrecisions = {
        // Convert layer will be inserted because of FP32 output, that allows:
        // - Propagate Dequantize through the Clamp, since if there is Return after the Clamp, then we cannot do
        // it(E#35846)
        // - Avoid an error in ngraph::float16::ie_abs (C#101214)
        InferenceEngine::Precision::FP32};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(inPrecisions), ::testing::ValuesIn(outrecisions),
                                           ::testing::ValuesIn(outFQAndClampRanges),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice));

INSTANTIATE_TEST_SUITE_P(precommit_QuantizedConvClamp, VPUXQuantizedConvClampSubGraphTest_VPU3720, basicCases,
                         VPUXQuantizedConvClampSubGraphTest::getTestCaseName);

}  // namespace
