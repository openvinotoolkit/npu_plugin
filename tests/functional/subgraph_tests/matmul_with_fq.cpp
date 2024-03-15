//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "vpu_ov2_layer_test.hpp"

namespace ov::test::subgraph {

class VPUXMatMulWithFQTest : public VpuOv2LayerTest {
public:
    void generate_inputs(const std::vector<ov::Shape>& inputShapes) override {
        VPUX_THROW_UNLESS(inputShapes.size() == 1, "Only 1 input shape is supported");
        const auto& funcInputs = function->inputs();
        VPUX_THROW_UNLESS(funcInputs.size() == 1, "Only 1 input is supported");
        const auto& inputStaticShape = inputShapes[0];
        const auto totalSize =
                std::accumulate(inputStaticShape.begin(), inputStaticShape.end(), 1, std::multiplies<size_t>());
        auto inputTensor = ov::Tensor{ov::element::f32, inputStaticShape};
        auto inputData = inputTensor.data<ov::element_type_traits<ov::element::f32>::value_type>();
        for (size_t i = 0; i < totalSize; i++) {
            inputData[i] = std::sin(i);
        }
        inputs = {
                {funcInputs[0].get_node_shared_ptr(), inputTensor},
        };
    }

    void compare(const std::vector<ov::Tensor>& expectedTensors,
                 const std::vector<ov::Tensor>& actualTensors) override {
        ASSERT_EQ(actualTensors.size(), 1);
        ASSERT_EQ(expectedTensors.size(), 1);

        const auto expected = expectedTensors[0];
        const auto actual = actualTensors[0];
        ASSERT_EQ(expected.get_size(), actual.get_size());

        const float absThreshold = 0.5f;
        ov::test::utils::compare(actual, expected, absThreshold);
    }

    void SetUp() override {
        // Create a subgraph that will be lowered into a FakeQuantize with two axes
        // FQ (3x32x64 * 3x1x64) -> Reshape (3x32x64 to 96x64) -> MatMul (16x96 * 96x64)
        constexpr size_t batchSize = 16;
        constexpr size_t numGroups = 3;
        constexpr size_t numColumns = 32;
        constexpr size_t numRows = 64;
        const std::vector<ov::Shape> inferenceShapes = {{batchSize, numColumns * numGroups}};
        const ov::test::InputShape dataShape = {{batchSize, numColumns * numGroups}, inferenceShapes};
        init_input_shapes({dataShape});
        const auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, inputDynamicShapes.at(0));
        const auto weightShape = ov::Shape{numGroups, numColumns, numRows};
        const auto weightTotalSize =
                std::accumulate(weightShape.cbegin(), weightShape.cend(), 1, std::multiplies<size_t>());
        std::vector<int8_t> weightsData(weightTotalSize, 0);
        for (size_t i = 0; i < weightsData.size(); i++) {
            weightsData.at(i) = i % 127;
        }
        const auto weights = ov::opset1::Constant::create(ov::element::i8, weightShape, weightsData);
        const auto convert = std::make_shared<ov::opset1::Convert>(weights->output(0), ov::element::f32);

        const auto scaleShiftShape = ov::Shape{numGroups, 1, numRows};
        const auto scaleShiftTotalSize =
                std::accumulate(scaleShiftShape.cbegin(), scaleShiftShape.cend(), 1, std::multiplies<size_t>());
        using scaleShiftValueType = ov::element_type_traits<ov::element::f32>::value_type;
        const std::vector<scaleShiftValueType> zeroPointData(scaleShiftTotalSize, 1);
        const auto zeroPoints = ov::opset1::Constant::create(ov::element::f32, scaleShiftShape, zeroPointData);
        const auto shift = std::make_shared<ov::opset1::Subtract>(convert->output(0), zeroPoints->output(0));

        std::vector<scaleShiftValueType> scaleData(scaleShiftTotalSize, 0);
        for (size_t i = 0; i < scaleData.size(); i++) {
            scaleData.at(i) = ((i % 7) + 2) / 128.f;
        }
        const auto scales = ov::opset1::Constant::create(ov::element::f32, scaleShiftShape, scaleData);
        const auto mul = std::make_shared<ov::opset1::Multiply>(shift->output(0), scales->output(0));

        const std::vector<int64_t> matrixShape = {numColumns * numGroups, numRows};
        const auto targetShape =
                ov::opset1::Constant::create(ov::element::i64, ov::Shape{matrixShape.size()}, matrixShape);
        const auto reshape = std::make_shared<ov::opset1::Reshape>(mul->output(0), targetShape->output(0), false);

        const auto matmul = std::make_shared<ov::opset1::MatMul>(param->output(0), reshape->output(0), false, false);

        const auto results = ov::ResultVector{std::make_shared<ov::opset1::Result>(matmul->output(0))};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "MatMulWithFQ");
    }
};

//
// Platform test definition
//

TEST_F(VPUXMatMulWithFQTest, NPU3720) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3720);
}

}  // namespace ov::test::subgraph
