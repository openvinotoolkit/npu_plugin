//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux_layer_test.hpp"

#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/single_layer/eltwise.hpp>

namespace ov::test::subgraph {

class VPUXEltwiseSameInputTest : public EltwiseLayerTest, virtual public VPUXLayerTest {};

TEST_P(VPUXEltwiseSameInputTest, DISABLED_VPU3700_HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3700);
}

TEST_P(VPUXEltwiseSameInputTest, VPU3720_HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3720);
}

}  // namespace ov::test::subgraph

namespace {

using namespace ov::test::subgraph;

std::vector<std::vector<ov::Shape>> inShapes = {
        {{1, 16, 128, 128}},
};

INSTANTIATE_TEST_CASE_P(
        smoke_EltwiseSameInput, VPUXEltwiseSameInputTest,
        ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                           ::testing::Values(ngraph::helpers::EltwiseTypes::ADD),
                           ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                           ::testing::Values(CommonTestUtils::OpType::VECTOR), ::testing::Values(ov::element::f16),
                           ::testing::Values(ov::element::f16), ::testing::Values(ov::element::f16),
                           ::testing::Values(targetDevice), ::testing::Values(ov::test::Config{})),
        VPUXEltwiseSameInputTest::getTestCaseName);

}  // namespace
