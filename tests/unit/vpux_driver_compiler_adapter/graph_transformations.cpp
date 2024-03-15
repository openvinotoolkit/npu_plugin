//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "graph_transformations.h"
#include "vpux_driver_compiler_adapter.h"

#include <gtest/gtest.h>

#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset6.hpp>

using namespace vpux::driverCompilerAdapter;

class GraphTransformations_UnitTests : public ::testing::Test {
protected:
    std::shared_ptr<ov::Model> opset6mvn;

    void SetUp() override;
};

void GraphTransformations_UnitTests::SetUp() {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3, 4});
    const auto axesConst = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
    const auto mvn = std::make_shared<ov::opset6::MVN>(data, axesConst, false, 1e-5, ov::op::MVNEpsMode::OUTSIDE_SQRT);

    opset6mvn = std::make_shared<ov::Model>(ov::NodeVector{mvn}, ov::ParameterVector{data});
}

//------------------------------------------------------------------------------
using GraphTransformations_Serialize = GraphTransformations_UnitTests;

TEST_F(GraphTransformations_Serialize, canSerializeToIR) {
    ASSERT_NO_THROW(graphTransformations::serializeToIR(opset6mvn));
}

TEST_F(GraphTransformations_Serialize, resultOfSerializationIsNotEmpy) {
    const IR ir = graphTransformations::serializeToIR(opset6mvn);

    EXPECT_GT(ir.xml.size(), 0);
    EXPECT_GT(ir.weights.size(), 0);
}

//------------------------------------------------------------------------------
using GraphTransformations_isFuncSupported = GraphTransformations_UnitTests;

TEST_F(GraphTransformations_isFuncSupported, opset6Function_forOpset5Compiler_NotSupported) {
    const std::string opset = "opset5";
    const bool isSupported = graphTransformations::isFunctionSupported(opset6mvn, opset);
    ASSERT_FALSE(isSupported);
}

TEST_F(GraphTransformations_isFuncSupported, opset6Function_forOpset6Compiler_IsSupported) {
    const std::string opset = "opset6";
    const bool isSupported = graphTransformations::isFunctionSupported(opset6mvn, opset);
    ASSERT_TRUE(isSupported);
}

TEST_F(GraphTransformations_isFuncSupported, opset6ParameterAndConstant_SupportedWithoutLowering) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 2, 3, 4});
    const auto mvn = std::make_shared<ov::opset4::MVN>(data, true, false, 1e-5);

    std::shared_ptr<ov::Model> opset4mvn_opset6params =
            std::make_shared<ov::Model>(ov::NodeVector{mvn}, ov::ParameterVector{data});

    const std::string supportedOpset = "opset4";
    const bool isSupported = graphTransformations::isFunctionSupported(opset4mvn_opset6params, supportedOpset);

    EXPECT_TRUE(isSupported);
}
