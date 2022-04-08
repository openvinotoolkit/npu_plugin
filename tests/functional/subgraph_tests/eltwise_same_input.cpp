//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "vpux_layer_test.hpp"

#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/single_layer/eltwise.hpp>

namespace {

class EltwiseSameInputTest :
        public ov::test::subgraph::EltwiseLayerTest,
        virtual public VPUXLayerTestsUtils::VPUXLayerTestsCommon {};

TEST_P(EltwiseSameInputTest, MLIR_HW) {
    useCompilerMLIR();
    setDefaultHardwareModeMLIR();
    run();
}

std::vector<std::vector<ov::Shape>> inShapes = {
        {{1, 16, 128, 128}},
};

INSTANTIATE_TEST_CASE_P(
        smoke, EltwiseSameInputTest,
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::ADD),
                           ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                           ::testing::Values(CommonTestUtils::OpType::VECTOR), ::testing::Values(ov::element::f16),
                           ::testing::Values(ov::element::f16), ::testing::Values(ov::element::f16),
                           ::testing::Values(VPUXLayerTestsUtils::testPlatformTargetDevice),
                           ::testing::Values(ov::test::Config{})),
        EltwiseSameInputTest::getTestCaseName);

}  // namespace
