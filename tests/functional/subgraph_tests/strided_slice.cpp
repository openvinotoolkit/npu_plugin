//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vpu_ov2_layer_test.hpp>

#include <ov_models/builders.hpp>
#include <ov_models/utils/ov_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace ov::test {

struct StridedSliceTestParams {
    ov::Shape _in_dims;
    std::vector<int64_t> _begin_data;
    std::vector<int64_t> _end_data;
    std::vector<int64_t> _strides_data;

    std::vector<int64_t> _begin_mask;
    std::vector<int64_t> _end_mask;
    std::vector<int64_t> _new_axis_mask;
    std::vector<int64_t> _shrink_axis_mask;
    std::vector<int64_t> _ellipsis_mask;
};

class StridedSliceSubGraphTest_NPU3700 :
        public VpuOv2LayerTest,
        public testing::WithParamInterface<StridedSliceTestParams> {
    void SetUp() override {
        const auto test_params = GetParam();
        const ov::Shape inputShape = test_params._in_dims;
        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes.front())};

        std::vector<int64_t> sliceBegin = test_params._begin_data;
        const auto sliceBeginConst =
                ngraph::builder::makeConstant<int64_t>(ov::element::i64, {sliceBegin.size()}, sliceBegin, false);
        std::vector<int64_t> sliceEnd = test_params._end_data;
        const auto sliceEndConst =
                ngraph::builder::makeConstant<int64_t>(ov::element::i64, {sliceEnd.size()}, sliceEnd, false);
        std::vector<int64_t> sliceStrides = test_params._strides_data;
        const auto sliceStridesConst =
                ngraph::builder::makeConstant<int64_t>(ov::element::i64, {sliceStrides.size()}, sliceStrides, false);

        const auto stridedSlice = std::make_shared<ov::op::v1::StridedSlice>(
                params.at(0), sliceBeginConst, sliceEndConst, sliceStridesConst, test_params._begin_mask,
                test_params._end_mask, test_params._new_axis_mask, test_params._shrink_axis_mask,
                test_params._ellipsis_mask);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(stridedSlice)};

        function = std::make_shared<ov::Model>(results, params, "KmbStridedSliceSubGraphTest");
        rel_threshold = 0.5f;
    }
};

TEST_P(StridedSliceSubGraphTest_NPU3700, HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3700);
}

INSTANTIATE_TEST_SUITE_P(smoke_StridedSlice, StridedSliceSubGraphTest_NPU3700,
                         ::testing::Values(
                                 StridedSliceTestParams{
                                         {1, 3, 64, 64},  // in dims
                                         {0, 0, 0, 0},    // begin data
                                         {1, 3, 64, 64},  // end data
                                         {1, 1, 2, 2},    // strides data
                                         {0, 0, 1, 1},    // begin mask
                                         {1, 0, 1, 1},    // end mask
                                         {0, 0, 0, 0},    // new axis mask
                                         {0, 0, 0, 0},    // shrink axis mask
                                         {0, 0, 0, 0},    // ellipsis mask
                                 },
                                 StridedSliceTestParams{
                                         {1, 3, 64, 64},  // in dims
                                         {0, 0, 0, 16},   // begin data
                                         {1, 3, 64, 32},  // end data
                                         {1, 1, 2, 2},    // strides data
                                         {1, 1, 0, 1},    // begin mask
                                         {1, 1, 0, 1},    // end mask
                                         {0, 0, 0, 0},    // new axis mask
                                         {0, 0, 0, 0},    // shrink axis mask
                                         {0, 0, 0, 0},    // ellipsis mask
                                 }));

}  // namespace ov::test
