// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <ov_models/builders.hpp>
#include <ov_models/utils/ov_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace ov::test {

class ConvClampSubGraphTest_NPU3700 : public VpuOv2LayerTest {
    void SetUp() override {
        const ov::Shape inputShape{1, 3, 62, 62};
        const ov::Shape weightsShape{48, 3, 3, 3};

        const auto weights = ngraph::builder::makeConstant<uint8_t>(ov::element::f32, weightsShape, {2}, false);

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params = {
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        const ov::Strides strides = {1, 1};
        const ov::CoordinateDiff pads_begin = {0, 0};
        const ov::CoordinateDiff pads_end = {0, 0};
        const ov::Strides dilations = {1, 1};
        const auto conv =
                std::make_shared<ov::op::v1::Convolution>(params[0], weights, strides, pads_begin, pads_end, dilations);

        const auto clamp = std::make_shared<ov::op::v0::Clamp>(conv, -1.0f, 1.0f);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(clamp)};
        function = std::make_shared<ov::Model>(results, params, "ConvClamp");
        rel_threshold = 0.1f;
    }
};

TEST_F(ConvClampSubGraphTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3700);
}
}  // namespace ov::test
