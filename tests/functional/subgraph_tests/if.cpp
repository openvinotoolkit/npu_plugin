//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpu_ov2_layer_test.hpp>
#include "subgraph_tests/nce_tasks.hpp"

#include <ov_models/builders.hpp>
#include <ov_models/utils/ov_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

using namespace ov;
using namespace element;

namespace ov::test {

class IfTestCommon : public VpuOv2LayerTest, public testing::WithParamInterface<std::tuple<Type, ov::Shape>> {
    void SetUp() override {
        inType = outType = std::get<0>(GetParam());
        const auto inputShape = std::get<1>(GetParam());

        init_input_shapes(ov::test::static_shapes_to_test_representation({Shape{1}, inputShape, inputShape}));

        auto cond = std::make_shared<op::v0::Parameter>(ov::element::i8, inputDynamicShapes[0]);
        auto X = std::make_shared<op::v0::Parameter>(inType, inputDynamicShapes[1]);
        auto Y = std::make_shared<op::v0::Parameter>(inType, inputDynamicShapes[2]);

        auto Xt = std::make_shared<op::v0::Parameter>(inType, PartialShape::dynamic());
        auto Yt = std::make_shared<op::v0::Parameter>(inType, PartialShape::dynamic());
        auto Xe = std::make_shared<op::v0::Parameter>(inType, PartialShape::dynamic());
        auto Ye = std::make_shared<op::v0::Parameter>(inType, PartialShape::dynamic());

        auto then_op_1 = std::make_shared<op::v1::Power>(Xt, Yt);
        auto then_op_2 = std::make_shared<op::v1::Multiply>(then_op_1, Yt);
        auto then_op_3 = std::make_shared<op::v1::Multiply>(then_op_2, Yt);
        auto else_op_1 = std::make_shared<op::v1::Add>(Xe, Ye);
        auto else_op_2 = std::make_shared<op::v1::Power>(else_op_1, Ye);
        auto then_op_result_1 = std::make_shared<op::v0::Result>(then_op_2);
        auto then_op_result_2 = std::make_shared<op::v0::Result>(then_op_3);
        auto else_op_result_1 = std::make_shared<op::v0::Result>(else_op_1);
        auto else_op_result_2 = std::make_shared<op::v0::Result>(else_op_2);
        auto then_body =
                std::make_shared<ov::Model>(OutputVector{then_op_result_1, then_op_result_2}, ParameterVector{Xt, Yt});
        auto else_body =
                std::make_shared<ov::Model>(OutputVector{else_op_result_1, else_op_result_2}, ParameterVector{Xe, Ye});
        auto if_op = std::make_shared<op::v8::If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xt, Xe);
        if_op->set_input(Y, Yt, Ye);

        auto rs1 = if_op->set_output(then_op_result_1, else_op_result_1);
        auto rs2 = if_op->set_output(then_op_result_2, else_op_result_2);

        auto result1 = std::make_shared<op::v0::Result>(rs1);
        auto result2 = std::make_shared<op::v0::Result>(rs2);

        function = std::make_shared<ov::Model>(OutputVector{result1, result2}, ParameterVector{cond, X, Y}, "IfTest");
    }
};

class IfTest_NPU3720 : public IfTestCommon {};

TEST_P(IfTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3720);
}

TEST_P(IfTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3720);
}

const TypeVector inType = {
        element::f16,
};

const std::vector<ov::Shape> inputShapes = {{1, 1, 4, 4}, {1, 2, 32, 64}};

INSTANTIATE_TEST_SUITE_P(smoke_IfTest, IfTest_NPU3720,
                         ::testing::Combine(::testing::ValuesIn(inType), ::testing::ValuesIn(inputShapes)));

}  // namespace ov::test
