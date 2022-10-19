//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

//
//          [Conv]
//             |
//       [FakeQuantize]
//             |
//          [Deconv] --- [FakeQuantize] -- [DeclareOp]
//             |
//          [output]
//

struct KmbDeconvFQTestParams {
    LayerTestsUtils::TargetDevice device;
    InferenceEngine::SizeVector in_dims;
    InferenceEngine::SizeVector w_dims;
    std::vector<uint64_t> conv_strides;
    std::vector<int64_t> conv_pads_begin;
    std::vector<int64_t> conv_pads_end;
    std::vector<uint64_t> deconv_strides;
    std::vector<int64_t> deconv_pads_begin;
    std::vector<int64_t> deconv_pads_end;
    std::vector<int64_t> deconv_out_pads;
};

class KmbDeconvFQTest :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<KmbDeconvFQTestParams> {
    void SetUp() override {
        const auto test_params = GetParam();
        targetDevice = test_params.device;
        const InferenceEngine::SizeVector inputShape = test_params.in_dims;
        const InferenceEngine::SizeVector weightsShape = test_params.w_dims;

        const auto params = ngraph::builder::makeParams(ngraph::element::f16, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        // Convolution
        const auto weightsConv =
                ngraph::builder::makeConstant<float>(ngraph::element::f16, weightsShape, {2.0f}, false);
        const ngraph::Strides stridesConv = test_params.conv_strides;
        const ngraph::CoordinateDiff padsBeginConv = test_params.conv_pads_begin;
        const ngraph::CoordinateDiff padsEndConv = test_params.conv_pads_end;
        const ngraph::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ngraph::opset2::Convolution>(paramOuts[0], weightsConv, stridesConv,
                                                                        padsBeginConv, padsEndConv, dilations);

        // ConvFQ
        const size_t dataLevels = 256;
        const std::vector<float> dataInputLow = {0.0f};
        const std::vector<float> dataInputHigh = {1.0f};
        const auto convFQ = ngraph::builder::makeFakeQuantize(conv, ngraph::element::f32, dataLevels, {}, dataInputLow,
                                                              dataInputHigh, dataInputLow, dataInputHigh);

        // Weights for Deconv
        const auto weightsDeconv =
                ngraph::builder::makeConstant<float>(ngraph::element::f16, weightsShape, {-1.0f}, false);

        // WeightsFQ
        const std::vector<float> dataWeightsLow = {0.0f};
        const std::vector<float> dataWeightsHigh = {100.0f};
        const auto weightsFQ =
                ngraph::builder::makeFakeQuantize(weightsDeconv, ngraph::element::f16, dataLevels, {}, dataWeightsLow,
                                                  dataWeightsHigh, dataWeightsLow, dataWeightsHigh);

        // Deconvolution
        const ngraph::Strides stridesDeconv = test_params.deconv_strides;
        const ngraph::CoordinateDiff padsBeginDeconv = test_params.deconv_pads_begin;
        const ngraph::CoordinateDiff padsEndDeconv = test_params.deconv_pads_end;
        const ngraph::CoordinateDiff outputPaddingDeconv = test_params.deconv_out_pads;
        const auto autoPadDeconv = ngraph::op::PadType::EXPLICIT;
        auto deconv2d_node = std::make_shared<ngraph::op::v1::ConvolutionBackpropData>(
                convFQ, weightsFQ, stridesDeconv, padsBeginDeconv, padsEndDeconv, dilations, autoPadDeconv,
                outputPaddingDeconv);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(deconv2d_node)};

        function = std::make_shared<ngraph::Function>(results, params, "KmbDeconvFQTest");
        threshold = 0.5f;
    }
};

TEST_P(KmbDeconvFQTest, CompareWithRefs_MLIR) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke, KmbDeconvFQTest,
    ::testing::Values(
    KmbDeconvFQTestParams {
        LayerTestsUtils::testPlatformTargetDevice,  // device
        {1, 3, 8, 14},     // in dims
        {16, 3, 4, 4},     // weights dims
        {1, 1},            // conv_strides
        {0, 0},            // conv_pads_begin
        {0, 0},            // conv_pads_end
        {4, 4},            // deconv_strides
        {2, 2},            // deconv_pads_begin
        {2, 2},            // deconv_pads_end
        {0, 0}             // deconv_output_padding
    })
);
}  // namespace
