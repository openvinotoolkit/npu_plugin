//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include <memory>

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/init_node_info.hpp>

#include "vpux/compiler/dialect/IE/ops.hpp"
#include "vpux/compiler/frontend/IE.hpp"

struct StridedSliceInputParams {
    InferenceEngine::SizeVector inputShape;
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> strides;
    std::vector<int64_t> beginMask;
    std::vector<int64_t> endMask;
    std::vector<int64_t> newAxisMask;
    std::vector<int64_t> shrinkAxisMask;
    std::vector<int64_t> ellipsisAxisMask;
};

class StridedSliceLayerTest : public testing::TestWithParam<StridedSliceInputParams> {};

static std::shared_ptr<ngraph::opset1::Constant> makeConstant1D(const std::vector<int64_t>& vec) {
    if (vec.size() == 0)
        return std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{});
    return std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{vec.size()}, vec);
}

TEST_P(StridedSliceLayerTest, TestWithParam) {
    std::shared_ptr<ngraph::Function> f;
    {
        auto p = this->GetParam();

        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape(p.inputShape));
        auto begin = makeConstant1D(p.begin);
        auto end = makeConstant1D(p.end);
        auto stride = makeConstant1D(p.strides);

        auto stridedSlice = std::make_shared<ngraph::opset1::StridedSlice>(
                data, begin, end, stride, p.beginMask, p.endMask, p.newAxisMask, p.shrinkAxisMask, p.ellipsisAxisMask);
        stridedSlice->set_friendly_name("StridedSlice");
        auto result = std::make_shared<ngraph::op::Result>(stridedSlice);
        f = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{data});

        ngraph::pass::InitNodeInfo().run_on_function(f);
    }
    InferenceEngine::CNNNetwork nGraphImpl(f);

    mlir::MLIRContext ctx;

    ctx.loadDialect<vpux::IE::IEDialect>();
    ctx.loadDialect<mlir::StandardOpsDialect>();

    EXPECT_NO_THROW(vpux::IE::importNetwork(&ctx, nGraphImpl, true));
}

const auto strided_slice_import_network_params = ::testing::Values(StridedSliceInputParams{
        {128, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {0, 1, 1}, {0, 1, 1}, {1, 0, 0}, {0, 0, 0}, {0, 0, 0}});

std::vector<StridedSliceInputParams> strided_slice_test_cases = {
        StridedSliceInputParams{
                {128, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {0, 1, 1}, {0, 1, 1}, {1, 0, 0}, {0, 0, 0}, {0, 0, 0}},
        StridedSliceInputParams{
                {128, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1}, {1, 0, 1}, {1, 0, 1}, {0, 1, 0}, {0, 0, 0}, {0, 0, 0}},
        StridedSliceInputParams{
                {1, 12, 100}, {0, -1, 0}, {0, 0, 0}, {1, 1, 1}, {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}},
        StridedSliceInputParams{
                {1, 12, 100}, {0, 9, 0}, {0, 11, 0}, {1, 1, 1}, {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
        StridedSliceInputParams{
                {1, 12, 100}, {0, 1, 0}, {0, -1, 0}, {1, 1, 1}, {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
        StridedSliceInputParams{{1, 12, 100},
                                {0, 9, 0},
                                {0, 7, 0},
                                {-1, -1, -1},
                                {1, 0, 1},
                                {1, 0, 1},
                                {0, 0, 0},
                                {0, 0, 0},
                                {0, 0, 0}},
        StridedSliceInputParams{
                {1, 12, 100}, {0, 7, 0}, {0, 9, 0}, {-1, 1, -1}, {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
        StridedSliceInputParams{
                {1, 12, 100}, {0, 4, 0}, {0, 9, 0}, {-1, 2, -1}, {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
        StridedSliceInputParams{{1, 12, 100},
                                {0, 4, 0},
                                {0, 10, 0},
                                {-1, 2, -1},
                                {1, 0, 1},
                                {1, 0, 1},
                                {0, 0, 0},
                                {0, 0, 0},
                                {0, 0, 0}},
        StridedSliceInputParams{{1, 12, 100},
                                {0, 9, 0},
                                {0, 4, 0},
                                {-1, -2, -1},
                                {1, 0, 1},
                                {1, 0, 1},
                                {0, 0, 0},
                                {0, 0, 0},
                                {0, 0, 0}},
        StridedSliceInputParams{{1, 12, 100},
                                {0, 10, 0},
                                {0, 4, 0},
                                {-1, -2, -1},
                                {1, 0, 1},
                                {1, 0, 1},
                                {0, 0, 0},
                                {0, 0, 0},
                                {0, 0, 0}},
        StridedSliceInputParams{{1, 12, 100},
                                {0, 11, 0},
                                {0, 0, 0},
                                {-1, -2, -1},
                                {1, 0, 1},
                                {1, 0, 1},
                                {0, 0, 0},
                                {0, 0, 0},
                                {0, 0, 0}},
        StridedSliceInputParams{{1, 12, 100},
                                {0, -6, 0},
                                {0, -8, 0},
                                {-1, -2, -1},
                                {1, 0, 1},
                                {1, 0, 1},
                                {0, 0, 0},
                                {0, 0, 0},
                                {0, 0, 0}},
        StridedSliceInputParams{{1, 12, 100, 1, 1},
                                {0, -1, 0, 0},
                                {0, 0, 0, 0},
                                {1, 1, 1, 1},
                                {1, 0, 1, 0},
                                {1, 0, 1, 0},
                                {},
                                {0, 1, 0, 1},
                                {}},
        StridedSliceInputParams{
                {2, 2, 2, 2}, {0, 0, 0, 0}, {2, 2, 2, 2}, {1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}},
        StridedSliceInputParams{
                {2, 2, 2, 2}, {1, 1, 1, 1}, {2, 2, 2, 2}, {1, 1, 1, 1}, {0, 0, 0, 0}, {1, 1, 1, 1}, {}, {}, {}},
        StridedSliceInputParams{
                {2, 2, 2, 2}, {1, 1, 1, 1}, {2, 2, 2, 2}, {1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {}, {}, {}},
        StridedSliceInputParams{
                {2, 2, 4, 3}, {0, 0, 0, 0}, {2, 2, 4, 3}, {1, 1, 2, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}},
        StridedSliceInputParams{
                {2, 2, 4, 2}, {1, 0, 0, 1}, {2, 2, 4, 2}, {1, 1, 2, 1}, {0, 1, 1, 0}, {1, 1, 0, 0}, {}, {}, {}},
        StridedSliceInputParams{
                {1, 2, 4, 2}, {1, 0, 0, 0}, {1, 2, 4, 2}, {1, 1, -2, -1}, {1, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}},
        StridedSliceInputParams{
                {2, 2, 4, 2}, {1, 0, 0, 0}, {1, 2, 4, 2}, {1, 1, -2, -1}, {0, 1, 1, 1}, {1, 1, 1, 1}, {}, {}, {}},
        StridedSliceInputParams{{2, 3, 4, 5, 6},
                                {0, 1, 0, 0, 0},
                                {2, 3, 4, 5, 6},
                                {1, 1, 1, 1, 1},
                                {1, 0, 1, 1, 1},
                                {1, 0, 1, 1, 1},
                                {},
                                {0, 1, 0, 0, 0},
                                {}},
        StridedSliceInputParams{{10, 12}, {-1, 1}, {-9999, 0}, {-1, 1}, {0, 1}, {0, 1}, {0, 0}, {0, 0}, {0, 0}},
        StridedSliceInputParams{{5, 5, 5, 5},
                                {-1, 0, -1, 0},
                                {-50, 0, -60, 0},
                                {-1, 1, -1, 1},
                                {0, 0, 0, 0},
                                {0, 1, 0, 1},
                                {0, 0, 0, 0},
                                {0, 0, 0, 0},
                                {0, 0, 0, 0}},
};

INSTANTIATE_TEST_CASE_P(MLIR_IE_FrontEndTest, StridedSliceLayerTest, ::testing::ValuesIn(strided_slice_test_cases));
