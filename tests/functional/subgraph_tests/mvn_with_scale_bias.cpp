// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <shared_test_classes/base/layer_test_utils.hpp>
#include "ov_models/builders.hpp"

namespace ov::test {

class MVNWithScaleBiasTest_NPU3720 : public VpuOv2LayerTest, public testing::WithParamInterface<std::vector<int64_t>> {
    void SetUp() override {
        const ov::Shape inputShape = {1, 1, 1, 320};
        ov::Layout order = "NHWC";
        inType = outType = ov::element::f16;

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        auto acrossChanels = false;
        auto normalizeVariance = true;
        float epsilonF = 0.0001;
        auto mvn = std::dynamic_pointer_cast<ov::op::v0::MVN>(
                ngraph::builder::makeMVN(params[0], acrossChanels, normalizeVariance, epsilonF));

        const size_t scaleShiftSize = inputShape[3];
        std::vector<double> multiplyData(scaleShiftSize, 0.078740157480314959);
        const auto multiplyConst = std::make_shared<ov::op::v0::Constant>(
                ov::element::f32, ov::Shape{1, 1, 1, scaleShiftSize}, multiplyData);
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(mvn, multiplyConst);

        std::vector<float> biases(scaleShiftSize, 1.0);
        for (std::size_t i = 0; i < biases.size(); i++) {
            biases.at(i) = i * 0.25f;
        }
        auto bias_weights_node = std::make_shared<ov::op::v0::Constant>(
                ov::element::f32, ov::Shape{1, 1, 1, scaleShiftSize}, biases.data());
        auto bias_node = std::make_shared<ov::op::v1::Add>(multiply, bias_weights_node->output(0));

        const auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(bias_node);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(sigmoid)};

        function = std::make_shared<ov::Model>(results, params, "MVNWithScaleBias");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout(order);
        preProc.input().model().set_layout(order);
        preProc.output().tensor().set_layout(order);
        preProc.output().model().set_layout(order);
        function = preProc.build();
        rel_threshold = 0.95f;
    }
};

TEST_F(MVNWithScaleBiasTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3720);
}
}  // namespace ov::test
