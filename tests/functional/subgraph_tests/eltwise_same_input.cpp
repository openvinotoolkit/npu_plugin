//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <ov_models/utils/ov_helpers.hpp>
#include <shared_test_classes/single_op/eltwise.hpp>

using namespace ov::test::utils;
using namespace ov::test;

namespace {

class EltwiseSameInputTestCommon : public EltwiseLayerTest, public VpuOv2LayerTest {};

TEST_P(EltwiseSameInputTestCommon, DISABLED_NPU3700_HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3700);
}

TEST_P(EltwiseSameInputTestCommon, NPU3720_HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3720);
}

std::vector<std::vector<ov::Shape>> inShapes = {
        {{1, 16, 128, 128}},
};

INSTANTIATE_TEST_CASE_P(smoke_EltwiseSameInput, EltwiseSameInputTestCommon,
                        ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes)),
                                           ::testing::Values(EltwiseTypes::ADD),
                                           ::testing::Values(InputLayerType::PARAMETER),
                                           ::testing::Values(OpType::VECTOR), ::testing::Values(ov::element::f16),
                                           ::testing::Values(ov::element::f16), ::testing::Values(ov::element::f16),
                                           ::testing::Values(DEVICE_NPU), ::testing::Values(ov::test::Config{})),
                        EltwiseSameInputTestCommon::getTestCaseName);

}  // namespace
