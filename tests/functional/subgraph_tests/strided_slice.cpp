// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <shared_test_classes/base/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>

namespace {

struct StridedSliceTestParams {
    LayerTestsUtils::TargetDevice _device;
    InferenceEngine::SizeVector _in_dims;
    std::vector<int64_t> _begin_data;
    std::vector<int64_t> _end_data;
    std::vector<int64_t> _strides_data;

    std::vector<int64_t> _begin_mask;
    std::vector<int64_t> _end_mask;
    std::vector<int64_t> _new_axis_mask;
    std::vector<int64_t> _shrink_axis_mask;
    std::vector<int64_t> _ellipsis_mask;
};

class KmbStridedSliceSubGraphTest : public LayerTestsUtils::KmbLayerTestsCommon,
                                    public testing::WithParamInterface<StridedSliceTestParams> {
    void SetUp() override {
        const auto test_params = GetParam();
        targetDevice = test_params._device;;
        const InferenceEngine::SizeVector inputShape = test_params._in_dims;

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        std::vector<int64_t> sliceBegin = test_params._begin_data;
        const auto sliceBeginConst = ngraph::builder::makeConstant<int64_t>(ngraph::element::i64, {sliceBegin.size()}, sliceBegin, false);
        std::vector<int64_t> sliceEnd = test_params._end_data;
        const auto sliceEndConst = ngraph::builder::makeConstant<int64_t>(ngraph::element::i64, {sliceEnd.size()}, sliceEnd, false);
        std::vector<int64_t> sliceStrides = test_params._strides_data;
        const auto sliceStridesConst = ngraph::builder::makeConstant<int64_t>(ngraph::element::i64, {sliceStrides.size()}, sliceStrides, false);

        const auto stridedSlice =
            std::make_shared<ngraph::op::v1::StridedSlice>(paramOuts.at(0),
                                                           sliceBeginConst,
                                                           sliceEndConst,
                                                           sliceStridesConst,
                                                           test_params._begin_mask,
                                                           test_params._end_mask,
                                                           test_params._new_axis_mask,
                                                           test_params._shrink_axis_mask,
                                                           test_params._ellipsis_mask);
        const ngraph::ResultVector results{
            std::make_shared<ngraph::opset1::Result>(stridedSlice)
        };

        function = std::make_shared<ngraph::Function>(results, params, "KmbStridedSliceSubGraphTest");

        threshold = 0.5f;
    }
};

TEST_P(KmbStridedSliceSubGraphTest, CompareWithRefs) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke, KmbStridedSliceSubGraphTest,
    ::testing::Values(
    StridedSliceTestParams {
        LayerTestsUtils::testPlatformTargetDevice,  // _device
        {1, 3, 64, 64},    // in dims
        {0, 0, 0, 0},      // begin data
        {1, 3, 64, 64},    // end data
        {1, 1, 2, 2},      // strides data
        {0, 0, 1, 1},      // begin mask
        {1, 0, 1, 1},      // end mask
        {0, 0, 0, 0},      // new axis mask
        {0, 0, 0, 0},      // shrink axis mask
        {0, 0, 0, 0},      // ellipsis mask
    },
    StridedSliceTestParams {
        LayerTestsUtils::testPlatformTargetDevice,  // _device
        {1, 3, 64, 64},    // in dims
        {0, 0, 0, 16},     // begin data
        {1, 3, 64, 32},    // end data
        {1, 1, 2, 2},      // strides data
        {1, 1, 0, 1},      // begin mask
        {1, 1, 0, 1},      // end mask
        {0, 0, 0, 0},      // new axis mask
        {0, 0, 0, 0},      // shrink axis mask
        {0, 0, 0, 0},      // ellipsis mask
    })
);

}
