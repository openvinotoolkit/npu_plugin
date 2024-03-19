//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <ov_models/builders.hpp>
#include <ov_models/utils/ov_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace ov::test {

class QuantizedInputConversionTest_NPU3700 : public VpuOv2LayerTest {
    void SetUp() override {
        const ov::Shape inputShape{1, 3, 352, 352};
        const ov::Shape weightsShape{32, 3, 3, 3};

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f16, inputDynamicShapes.front())};

        const std::vector<float> dataLow = {-10.0f};
        const std::vector<float> dataHigh = {10.0f};

        const auto inputConvFq = ngraph::builder::makeFakeQuantize(params[0], ov::element::f16, 256, {}, dataLow,
                                                                   dataHigh, dataLow, dataHigh);

        const auto weightsU8 = ngraph::builder::makeConstant<uint8_t>(ov::element::u8, weightsShape, {}, true, 255, 1);
        const auto weightsFP16 = std::make_shared<ov::op::v0::Convert>(weightsU8, ov::element::f16);
        const std::vector<float> weightsDataLow = {0.0f};
        const std::vector<float> weightsDataHigh = {183.0f};

        const auto weightsFq = ngraph::builder::makeFakeQuantize(weightsFP16, ov::element::f16, 256, {}, weightsDataLow,
                                                                 weightsDataHigh, weightsDataLow, weightsDataHigh);

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        const auto conv = std::make_shared<ov::op::v1::Convolution>(inputConvFq, weightsFq, strides, pads_begin,
                                                                    pads_end, dilations);

        const std::vector<float> outputConvLow = {0.0f};
        const std::vector<float> outputConvHigh = {154.0f};

        const auto outputConvFq = ngraph::builder::makeFakeQuantize(conv, ov::element::f16, 256, {}, outputConvLow,
                                                                    outputConvHigh, outputConvLow, outputConvHigh);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outputConvFq)};
        function = std::make_shared<ov::Model>(results, params, "QuantizedConvWithInputConversion");
        rel_threshold = 0.1f;
    }
};

TEST_F(QuantizedInputConversionTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3700);
}

}  // namespace ov::test
