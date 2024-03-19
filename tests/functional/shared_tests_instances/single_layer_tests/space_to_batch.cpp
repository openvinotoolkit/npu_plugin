//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <ngraph/opsets/opset2.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_op_tests/space_to_batch.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test;

namespace LayerTestsDefinitions {

class SpaceToBatchLayerTestCommon : public SpaceToBatchLayerTest, virtual public VpuOv2LayerTest {};
class SpaceToBatchLayerTest_NPU3720 : public SpaceToBatchLayerTestCommon {};

TEST_P(SpaceToBatchLayerTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3720);
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

const std::vector<ov::element::Type> Precisions = {ov::element::f16};

const std::vector<std::vector<ov::Shape>> shapes = {{{1, 12, 10}}, {{2, 8, 8, 3}}, {{2, 8, 8, 3, 3}}};

const auto precommit_SpaceToBatch_3D = ::testing::Combine(
        ::testing::Values(std::vector<int64_t>{1, 1, 8}), ::testing::Values(std::vector<int64_t>{0, 0, 2}),
        ::testing::Values(std::vector<int64_t>{0, 0, 4}),
        ::testing::ValuesIn({static_shapes_to_test_representation({shapes[0]})}), ::testing::ValuesIn(Precisions),
        ::testing::Values(ov::test::utils::DEVICE_NPU));

const auto precommit_SpaceToBatch_4D = ::testing::Combine(
        ::testing::Values(std::vector<int64_t>{1, 6, 4, 1}), ::testing::Values(std::vector<int64_t>{0, 1, 0, 0}),
        ::testing::Values(std::vector<int64_t>{0, 3, 0, 0}),
        ::testing::ValuesIn({static_shapes_to_test_representation({shapes[1]})}), ::testing::ValuesIn(Precisions),
        ::testing::Values(ov::test::utils::DEVICE_NPU));

const auto precommit_SpaceToBatch_5D = ::testing::Combine(
        ::testing::Values(std::vector<int64_t>{1, 6, 4, 1, 1}), ::testing::Values(std::vector<int64_t>{0, 1, 0, 0, 0}),
        ::testing::Values(std::vector<int64_t>{0, 3, 0, 0, 0}),
        ::testing::ValuesIn({static_shapes_to_test_representation({shapes[2]})}), ::testing::ValuesIn(Precisions),
        ::testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToBatch_3D_NPU3720, SpaceToBatchLayerTest_NPU3720,
                         precommit_SpaceToBatch_3D, SpaceToBatchLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToBatch_4D_NPU3720, SpaceToBatchLayerTest_NPU3720,
                         precommit_SpaceToBatch_4D, SpaceToBatchLayerTest_NPU3720::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_precommit_SpaceToBatch_5D_NPU3720, SpaceToBatchLayerTest_NPU3720,
                         precommit_SpaceToBatch_5D, SpaceToBatchLayerTest_NPU3720::getTestCaseName);

}  // namespace
