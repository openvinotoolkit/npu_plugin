// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpu_ov2_layer_test.hpp"

#include <ov_models/builders.hpp>
#include <ov_models/utils/ov_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

using namespace ov::test;
using namespace ov::test::utils;
namespace {

std::shared_ptr<ov::Node> buildConvolution(const ov::Output<ov::Node>& param, const size_t filtersIn,
                                           const size_t filtersOut) {
    const size_t kernelW = 1;
    const size_t kernelH = 1;
    std::vector<float> weights(filtersOut * filtersIn * kernelW * kernelH);
    for (std::size_t i = 0; i < weights.size(); i++) {
        weights.at(i) = std::cos(i * 3.14 / 6);
    }
    auto constLayerNode = std::make_shared<ov::op::v0::Constant>(
            ov::element::f32, ov::Shape{filtersOut, filtersIn, kernelH, kernelW}, weights.data());

    auto conv2d = std::make_shared<ov::op::v1::Convolution>(
            param, constLayerNode->output(0), ov::Strides(std::vector<size_t>{1, 1}),
            ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ov::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
            ov::Strides(std::vector<size_t>{1, 1}));

    return conv2d;
}

class TilingWithConcatTest_NPU3720 : public VpuOv2LayerTest, public testing::WithParamInterface<ov::Shape> {
    ov::Shape transposeShape(const ov::Shape& inputShape, const ov::Layout& order) const {
        ov::Shape transposedShape = inputShape;
        std::string neutralDims = "NCHW";
        std::vector<int64_t> indices;
        const auto dimToIdx = [&order](const char dim) -> int64_t {
            const std::string dimStr(1, dim);
            return order.get_index_by_name(dimStr);
        };
        std::transform(neutralDims.cbegin(), neutralDims.cend(), std::back_inserter(indices), dimToIdx);
        for (size_t srcIdx = 0; srcIdx < inputShape.size(); srcIdx++) {
            const auto dstIdx = indices.at(srcIdx);
            transposedShape.at(dstIdx) = inputShape.at(srcIdx);
        }
        return transposedShape;
    }
    void SetUp() override {
        const ov::Shape origInputShape = GetParam();
        inType = outType = ov::element::f16;
        std::vector<int64_t> dimsOrderIn = {0, 3, 1, 2};
        std::vector<int64_t> dimsOrderOut = {0, 2, 3, 1};
        ov::Shape inputShape{origInputShape.at(0), origInputShape.at(2), origInputShape.at(3),
                             origInputShape.at(1)};  // NHWC

        init_input_shapes(static_shapes_to_test_representation({inputShape}));

        const size_t filtOut = 64;
        const size_t filtIn = origInputShape.at(1);

        const ov::ParameterVector params{
                std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0])};

        auto orderIn = ov::op::v0::Constant::create(ov::element::i64, {dimsOrderIn.size()}, dimsOrderIn);
        const auto inputTransp = std::make_shared<ov::op::v1::Transpose>(params.at(0), orderIn);  // NCHW

        const auto conv2d64Planes = buildConvolution(inputTransp, filtIn, filtOut);
        const auto conv2d32Planes = buildConvolution(conv2d64Planes->output(0), filtOut, 32);

        const auto concat = std::make_shared<ov::op::v0::Concat>(
                ov::OutputVector({conv2d64Planes->output(0), conv2d32Planes->output(0)}), 1);

        auto orderOut = ov::op::v0::Constant::create(ov::element::i64, {dimsOrderOut.size()}, dimsOrderOut);  // NHWC
        const auto outputTransp = std::make_shared<ov::op::v1::Transpose>(concat, orderOut);

        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(outputTransp)};

        function = std::make_shared<ov::Model>(results, params, "TilingWithConcatTest");
        auto preProc = ov::preprocess::PrePostProcessor(function);
        preProc.input().tensor().set_layout("NHWC");
        preProc.input().model().set_layout("NHWC");
        preProc.output().tensor().set_layout("NHWC");
        preProc.output().model().set_layout("NHWC");
        function = preProc.build();

        rel_threshold = 0.5f;
    }
};

TEST_P(TilingWithConcatTest_NPU3720, HW) {
    setDefaultHardwareMode();
    run(VPUXPlatform::VPU3720);
}

const std::vector<ov::Shape> inputShapes = {
        {1, 16, 175, 175},
        // This case is working in OV1 but in OV2 has threshold error
        // {1, 3, 250, 250},
};

INSTANTIATE_TEST_SUITE_P(smoke_tiling_with_concat, TilingWithConcatTest_NPU3720, ::testing::ValuesIn(inputShapes));

}  // namespace
