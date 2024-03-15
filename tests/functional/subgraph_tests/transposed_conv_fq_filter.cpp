//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <ov_models/builders.hpp>
#include <ov_models/utils/ov_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace ov::test {

//
//          [Conv]
//             |
//       [FakeQuantize]
//             |
//      [TransposedConv] --- [FakeQuantize] -- [DeclareOp]
//             |
//          [output]
//

struct TransposedConvFQTestParams {
    ov::Shape in_dims;
    ov::Shape w_dims;
    std::vector<uint64_t> conv_strides;
    std::vector<int64_t> conv_pads_begin;
    std::vector<int64_t> conv_pads_end;
    std::vector<uint64_t> transposed_conv_strides;
    std::vector<int64_t> transposed_conv_pads_begin;
    std::vector<int64_t> transposed_conv_pads_end;
    std::vector<int64_t> transposed_conv_out_pads;
};

class TransposedConvFQTest_NPU3700 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<TransposedConvFQTestParams> {
    void SetUp() override {
        const auto test_params = GetParam();
        const ov::Shape inputShape = test_params.in_dims;
        const ov::Shape weightsShape = test_params.w_dims;

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        // Convolution
        const auto weightsConv = ngraph::builder::makeConstant<float>(ov::element::f16, weightsShape, {2.0f}, false);
        const ov::Strides stridesConv = test_params.conv_strides;
        const ov::CoordinateDiff padsBeginConv = test_params.conv_pads_begin;
        const ov::CoordinateDiff padsEndConv = test_params.conv_pads_end;
        const ov::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ov::op::v1::Convolution>(params[0], weightsConv, stridesConv, padsBeginConv,
                                                                    padsEndConv, dilations);

        // ConvFQ
        const size_t dataLevels = 256;
        const std::vector<float> dataInputLow = {0.0f};
        const std::vector<float> dataInputHigh = {1.0f};
        const auto convFQ = ngraph::builder::makeFakeQuantize(conv, ov::element::f32, dataLevels, {}, dataInputLow,
                                                              dataInputHigh, dataInputLow, dataInputHigh);

        // Weights for TransposedConv
        const auto weightsTransposedConv =
                ngraph::builder::makeConstant<float>(ov::element::f16, weightsShape, {-1.0f}, false);

        // WeightsFQ
        const std::vector<float> dataWeightsLow = {0.0f};
        const std::vector<float> dataWeightsHigh = {100.0f};
        const auto weightsFQ =
                ngraph::builder::makeFakeQuantize(weightsTransposedConv, ov::element::f16, dataLevels, {},
                                                  dataWeightsLow, dataWeightsHigh, dataWeightsLow, dataWeightsHigh);

        // TransposedConvolution
        const ov::Strides stridesTransposedConv = test_params.transposed_conv_strides;
        const ov::CoordinateDiff padsBeginTransposedConv = test_params.transposed_conv_pads_begin;
        const ov::CoordinateDiff padsEndTransposedConv = test_params.transposed_conv_pads_end;
        const ov::CoordinateDiff outputPaddingTransposedConv = test_params.transposed_conv_out_pads;
        const auto autoPadTransposedConv = ov::op::PadType::EXPLICIT;
        auto transposed_conv2d_node = std::make_shared<ov::op::v1::ConvolutionBackpropData>(
                convFQ, weightsFQ, stridesTransposedConv, padsBeginTransposedConv, padsEndTransposedConv, dilations,
                autoPadTransposedConv, outputPaddingTransposedConv);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(transposed_conv2d_node)};

        function = std::make_shared<ov::Model>(results, params, "TransposedConvFQTest");
        rel_threshold = 0.5f;
    }
};

TEST_P(TransposedConvFQTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3700);
}

INSTANTIATE_TEST_CASE_P(smoke_TransposedConvFQ, TransposedConvFQTest_NPU3700,
                        ::testing::Values(TransposedConvFQTestParams{
                                {1, 3, 8, 14},  // in dims
                                {16, 3, 4, 4},  // weights dims
                                {1, 1},         // conv_strides
                                {0, 0},         // conv_pads_begin
                                {0, 0},         // conv_pads_end
                                {4, 4},         // transposed_conv_strides
                                {2, 2},         // transposed_conv_pads_begin
                                {2, 2},         // transposed_conv_pads_end
                                {0, 0}          // transposed_conv_output_padding
                        }));
}  // namespace ov::test
