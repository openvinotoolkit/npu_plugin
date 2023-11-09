//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpu_ov1_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

#include <random>

namespace {

struct UpstreamReduceStridedSliceTestParams {
    LayerTestsUtils::TargetDevice _device;
    std::vector<InferenceEngine::SizeVector> _inDims;
    std::vector<ngraph::Shape> _filterDims;
    std::vector<std::vector<size_t>> _kernelStrides;
    std::vector<std::vector<int64_t>> _beginData;
    std::vector<std::vector<int64_t>> _endData;
    std::vector<std::vector<int64_t>> _beginMask;
    std::vector<std::vector<int64_t>> _endMask;
};

// Build following graph which exercises the possibility to
// upstream Slice_1 and later on fuse it with Slice_0
//
// ...  Conv --- > Slice_0 (H) --- >        < --- Reshape ...
//                                 |       |
//                                 v       v
//                                 Eltwise_0
//                                     |
//                                     V
//                                 Slice_1 (H)  --- > ...

class VPUXUpstreamReduceStridedSliceSubGraphTest_VPU3700 :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
        public testing::WithParamInterface<UpstreamReduceStridedSliceTestParams> {
    void ValidateTestParams(UpstreamReduceStridedSliceTestParams testParams) {
        ASSERT_EQ(testParams._inDims.size(), 2);
        ASSERT_EQ(testParams._filterDims.size(), 2);
        ASSERT_EQ(testParams._kernelStrides.size(), 2);
        ASSERT_EQ(testParams._beginData.size(), 2);
        ASSERT_EQ(testParams._endData.size(), 2);
        ASSERT_EQ(testParams._beginMask.size(), 2);
        ASSERT_EQ(testParams._endMask.size(), 2);
    }

    void SetUp() override {
        const auto testParams = GetParam();
        ValidateTestParams(testParams);

        targetDevice = testParams._device;

        const auto inputShapeArray = testParams._inDims;

        const auto paramIns = ngraph::builder::makeParams(ngraph::element::f32, inputShapeArray);
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(paramIns));

        std::mt19937 floatMersenneEngine{0};
        std::normal_distribution<double> floatDist{0.0, 1.0};
        auto floatGen = [&floatDist, &floatMersenneEngine]() {
            return floatDist(floatMersenneEngine);
        };

        std::mt19937 intMersenneEngine{0};
        std::normal_distribution<double> intDist{-127.0, 127.0};
        auto intGen = [&intDist, &intMersenneEngine]() {
            return (int8_t)std::round(intDist(intMersenneEngine));
        };

        std::vector<std::shared_ptr<ngraph::op::Op>> nodes;

        for (size_t idx = 0; idx < paramOuts.size(); idx++) {
            const auto filterShape = testParams._filterDims[idx];
            std::vector<int8_t> weightsData(ngraph::shape_size(filterShape));
            std::generate(weightsData.begin(), weightsData.end(), intGen);
            const auto weightsConst =
                    std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i8, filterShape, weightsData);

            const auto weightsConvert =
                    std::make_shared<ngraph::op::v0::Convert>(weightsConst, ngraph::element::Type_t::f32);

            std::vector<double> subtractData(filterShape[0], 0);
            const auto subtractConst = std::make_shared<ngraph::op::Constant>(
                    ngraph::element::Type_t::f32, ngraph::Shape({filterShape[0], 1, 1, 1}), subtractData);
            const auto weightsSubtract = std::make_shared<ngraph::op::v1::Subtract>(weightsConvert, subtractConst);

            std::vector<double> multiplyData(filterShape[0], 0.00784313725490196);
            const auto multiplyConst = std::make_shared<ngraph::op::Constant>(
                    ngraph::element::Type_t::f32, ngraph::Shape({filterShape[0], 1, 1, 1}), multiplyData);
            const auto weightsMultiply = std::make_shared<ngraph::op::v1::Multiply>(weightsSubtract, multiplyConst);

            const auto inputLowConst = std::make_shared<ngraph::op::Constant>(
                    ngraph::element::Type_t::f32, ngraph::Shape({1}), std::vector<double>({-1.0}));
            const auto inputHighConst = std::make_shared<ngraph::op::Constant>(
                    ngraph::element::Type_t::f32, ngraph::Shape({1}), std::vector<double>({1.0}));
            const auto inputFq = std::make_shared<ngraph::op::v0::FakeQuantize>(
                    paramOuts[idx], inputLowConst, inputHighConst, inputLowConst, inputHighConst, 256);

            const auto conv = std::make_shared<ngraph::op::v1::Convolution>(
                    inputFq, weightsMultiply, ngraph::Strides(testParams._kernelStrides[idx]),
                    ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                    ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                    ngraph::Strides(testParams._kernelStrides[idx]), ngraph::op::PadType::AUTO);

            const auto biasShape = ngraph::Shape({1, conv->get_output_shape(0)[1], 1, 1});
            std::vector<double> biasData(ngraph::shape_size(biasShape));
            std::generate(biasData.begin(), biasData.end(), floatGen);
            const auto biasConst =
                    std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::f32, biasShape, biasData);
            const auto bias = std::make_shared<ngraph::op::v1::Add>(conv, biasConst);

            const auto relu = std::make_shared<ngraph::op::v0::Relu>(bias);

            const auto actLowConst = std::make_shared<ngraph::op::Constant>(
                    ngraph::element::Type_t::f32, ngraph::Shape({1}), std::vector<double>({0.0}));
            const auto actHighConst = std::make_shared<ngraph::op::Constant>(
                    ngraph::element::Type_t::f32, ngraph::Shape({1}), std::vector<double>({4.0}));
            const auto activationFq = std::make_shared<ngraph::op::v0::FakeQuantize>(relu, actLowConst, actHighConst,
                                                                                     actLowConst, actHighConst, 256);

            nodes.push_back(activationFq);
        }

        const auto convSliceBeginConst = ngraph::builder::makeConstant<int64_t>(
                ngraph::element::i64, {testParams._beginData[0].size()}, testParams._beginData[0], false);
        const auto convSliceEndConst = ngraph::builder::makeConstant<int64_t>(
                ngraph::element::i64, {testParams._endData[0].size()}, testParams._endData[0], false);
        const auto convSliceStridesConst = ngraph::builder::makeConstant<int64_t>(
                ngraph::element::i64, {testParams._beginData[0].size()}, {1, 1, 1, 1}, false);

        const auto convStridedSlice = std::make_shared<ngraph::op::v1::StridedSlice>(
                nodes[0], convSliceBeginConst, convSliceEndConst, convSliceStridesConst, testParams._beginMask[0],
                testParams._endMask[0]);

        // Adjust shape by doubling width and halfing channels
        const auto reshapePattern =
                std::vector<int64_t>({0, -1, 0, static_cast<int64_t>(2 * nodes[1]->get_output_shape(0)[3])});
        const auto newShape =
                ngraph::builder::makeConstant<int64_t>(ngraph::element::i64, ngraph::Shape{4}, reshapePattern, false);
        const auto reshape = std::make_shared<ngraph::op::v1::Reshape>(nodes[1], newShape, true);

        const auto add = std::make_shared<ngraph::op::v1::Add>(convStridedSlice, reshape);

        const auto relu = std::make_shared<ngraph::op::v0::Relu>(add);

        const auto resultLowConst = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, ngraph::Shape({1}), std::vector<double>({0.0}));
        const auto resultHighConst = std::make_shared<ngraph::op::Constant>(
                ngraph::element::Type_t::f32, ngraph::Shape({1}), std::vector<double>({8.0}));

        const auto reluFq = std::make_shared<ngraph::op::v0::FakeQuantize>(relu, resultLowConst, resultHighConst,
                                                                           resultLowConst, resultHighConst, 256);

        const auto addSliceBeginConst = ngraph::builder::makeConstant<int64_t>(
                ngraph::element::i64, {testParams._beginData[1].size()}, testParams._beginData[1], false);
        const auto addSliceEndConst = ngraph::builder::makeConstant<int64_t>(
                ngraph::element::i64, {testParams._endData[1].size()}, testParams._endData[1], false);
        const auto addSliceStridesConst = ngraph::builder::makeConstant<int64_t>(
                ngraph::element::i64, {testParams._beginData[1].size()}, {1, 1, 1, 1}, false);

        const auto addStridedSlice = std::make_shared<ngraph::op::v1::StridedSlice>(
                reluFq, addSliceBeginConst, addSliceEndConst, addSliceStridesConst, testParams._beginMask[1],
                testParams._endMask[1]);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(addStridedSlice)};

        function = std::make_shared<ngraph::Function>(results, paramIns, "VPUXUpstreamReduceStridedSliceSubGraphTest");

        threshold = 0.5f;
    }
};

TEST_P(VPUXUpstreamReduceStridedSliceSubGraphTest_VPU3700, SW) {
    setPlatformVPU3700();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXUpstreamReduceStridedSliceSubGraphTest_VPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

INSTANTIATE_TEST_CASE_P(smoke_UpstreamReduceStridedSlice, VPUXUpstreamReduceStridedSliceSubGraphTest_VPU3700,
                        ::testing::Values(UpstreamReduceStridedSliceTestParams{
                                LayerTestsUtils::testPlatformTargetDevice(),  // device
                                {{1, 16, 64, 4}, {1, 16, 17, 2}},             // in dims
                                {{32, 16, 3, 1}, {64, 16, 3, 1}},             // filter dims
                                {{1, 1}, {1, 1}},                             // kernel strides
                                {{0, 0, 47, 0}, {0, 0, 1, 0}},                // begin data
                                {{0, 0, 0, 0}, {0, 0, 0, 0}},                 // end data
                                {{1, 1, 0, 1}, {1, 1, 0, 1}},                 // begin mask
                                {{1, 1, 1, 1}, {1, 1, 1, 1}}                  // end mask
                        }));

}  // namespace
