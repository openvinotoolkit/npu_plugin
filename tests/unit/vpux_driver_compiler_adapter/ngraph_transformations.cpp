//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//
#include <gtest/gtest.h>
#include "ngraph_transformations.h"
#include "vpux_driver_compiler_adapter.h"

#include "ngraph/ngraph.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include "ie_ngraph_utils.hpp"

using namespace vpux::driverCompilerAdapter;

class NgraphTransformations_UnitTests: public ::testing::Test {
protected:
    std::shared_ptr<ngraph::Function> opset6mvn;

    void SetUp() override;
};

void NgraphTransformations_UnitTests::SetUp() {
    const auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4 });
    const auto axesConst = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { 2, 3 });
    const auto mvn = std::make_shared<ngraph::opset6::MVN>(
            data, axesConst, false, 1e-5, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);

    opset6mvn = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ data });
}

//------------------------------------------------------------------------------
using NgraphTransformations_Serialize = NgraphTransformations_UnitTests;

TEST_F(NgraphTransformations_Serialize, canSerializeToIR) {
    ASSERT_NO_THROW(ngraphTransformations::serializeToIR(opset6mvn));
}

TEST_F(NgraphTransformations_Serialize, resultOfSerializationIsNotEmpy) {
    const IR ir = ngraphTransformations::serializeToIR(opset6mvn);

    EXPECT_GT(ir.xml.size(), 0);
    EXPECT_GT(ir.weights.size(), 0);
}

//------------------------------------------------------------------------------
using NgraphTransformations_isFuncSupported = NgraphTransformations_UnitTests;

TEST_F(NgraphTransformations_isFuncSupported, opset6Function_forOpset5Compiler_NotSupported) {
    const size_t opset = 5;
    const bool isSupported = ngraphTransformations::isFunctionSupported(opset6mvn, opset);
    ASSERT_FALSE(isSupported);
}

TEST_F(NgraphTransformations_isFuncSupported, opset6Function_forOpset6Compiler_IsSupported) {
    const size_t opset = 6;
    const bool isSupported = ngraphTransformations::isFunctionSupported(opset6mvn, opset);
    ASSERT_TRUE(isSupported);
}

TEST_F(NgraphTransformations_isFuncSupported, opset6ParameterAndConstant_SupportedWithoutLowering) {
    using parameterOpset = ngraph::opset6::Parameter;
    using mvnOpset = ngraph::opset4::MVN;

    const auto data = std::make_shared<parameterOpset>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4 });
    const auto mvn = std::make_shared<mvnOpset>(data, true, false, 1e-5);

    std::shared_ptr<ngraph::Function> opset4mvn_opset6params =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ data });

    const size_t supportedOpset = 4;
    const bool isSupported = ngraphTransformations::isFunctionSupported(opset4mvn_opset6params, supportedOpset);

    EXPECT_TRUE(isSupported);
}
