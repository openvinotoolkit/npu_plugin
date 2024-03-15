//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "single_op_tests/slice.hpp"
#include "vpu_ov2_layer_test.hpp"

using namespace ov::test;

namespace LayerTestsDefinitions {

class SliceLayerTestCommon : public Slice8LayerTest, virtual public VpuOv2LayerTest {};

class SliceLayerTest_NPU3720 : public SliceLayerTestCommon {};

TEST_P(SliceLayerTest_NPU3720, SW) {
    setReferenceSoftwareMode();
    run(VPUXPlatform::VPU3720);
}

}  // namespace LayerTestsDefinitions

using namespace LayerTestsDefinitions;

namespace {

std::vector<Slice8SpecificParams> staticParams = {

        Slice8SpecificParams{{{{}, {{16}}}}, {4}, {12}, {1}, {0}},
        Slice8SpecificParams{{{{}, {{20, 10}}}}, {0, 0}, {10, 20}, {1, 1}, {1, 0}},
        Slice8SpecificParams{{{{}, {{1, 12, 100}}}}, {0, 9, 0}, {1, 11, 1}, {1, 1, 1}, {0, 1, -1}},
        Slice8SpecificParams{{{{}, {{2, 30, 50}}}}, {0, 0, 4}, {-5, -1, -1}, {1, 2, 1}, {2, 0, 1}},
        Slice8SpecificParams{{{{}, {{16}}}}, {0}, {8}, {2}, {0}}};

const std::vector<ov::element::Type> netPrecisions = {ov::element::f16, ov::element::f32, ov::element::i32,
                                                      ov::element::u32, ov::element::u8,  ov::element::i8};

const auto sliceParams = testing::Combine(testing::ValuesIn(staticParams),   // params
                                          testing::ValuesIn(netPrecisions),  // Model type
                                          testing::Values(ov::test::utils::DEVICE_NPU));

INSTANTIATE_TEST_SUITE_P(smoke_Slice, SliceLayerTest_NPU3720, sliceParams, SliceLayerTest_NPU3720::getTestCaseName);

}  // namespace
