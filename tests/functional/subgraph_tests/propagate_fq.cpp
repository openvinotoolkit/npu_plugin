//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpu_ov1_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

class VPUXPropagateFQSubGraphTest_VPU3720 :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
        public testing::WithParamInterface<std::vector<int64_t>> {
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice();
        const InferenceEngine::SizeVector inputShape{1, 16, 32, 32};
        const auto transposeOrder = GetParam();
        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const size_t dataLevels = 256;
        const std::vector<float> inDataLow = {0.0f};
        const std::vector<float> inDataHigh = {100.0f};

        std::vector<size_t> newShape;
        newShape.push_back(inputShape[0]);
        newShape.push_back(inputShape[3]);
        newShape.push_back(inputShape[2]);
        newShape.push_back(inputShape[1]);
        const auto reshape = buildReshape(paramOuts[0], newShape);

        const auto lhsTranspose = buildTranspose(reshape, transposeOrder);

        const auto dataFq = ngraph::builder::makeFakeQuantize(lhsTranspose, ngraph::element::f32, dataLevels, {},
                                                              inDataLow, inDataHigh, inDataLow, inDataHigh);
        const ngraph::Strides strides = {2, 2};
        const std::vector<size_t> pads_begin = {0, 0};
        const std::vector<size_t> pads_end = {0, 0};
        const ngraph::Strides dilations = {1, 1};
        const std::vector<size_t> kernelSize = {2, 2};
        const ngraph::op::PadType padType = ngraph::op::PadType::AUTO;
        const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;

        const auto pooling =
                ngraph::builder::makePooling(dataFq, strides, pads_begin, pads_end, kernelSize, roundingType, padType,
                                             false, ngraph::helpers::PoolingTypes::MAX);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset7::Result>(pooling),
                                           std::make_shared<ngraph::opset7::Result>(lhsTranspose)};

        function = std::make_shared<ngraph::Function>(results, params, "VPUXPropagateFQSubGraph");
    }

    std::shared_ptr<ov::Node> buildTranspose(const ov::Output<ov::Node>& param, const std::vector<int64_t>& dimsOrder) {
        auto order = ngraph::opset8::Constant::create(ngraph::element::i64, {dimsOrder.size()}, dimsOrder);
        return std::make_shared<ngraph::opset7::Transpose>(param, order);
    }

    std::shared_ptr<ov::Node> buildReshape(const ov::Output<ov::Node>& param, const std::vector<size_t>& newShape) {
        auto constNode = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64,
                                                                    ngraph::Shape{newShape.size()}, newShape);
        const auto reshape = std::dynamic_pointer_cast<ngraph::opset1::Reshape>(
                std::make_shared<ngraph::opset1::Reshape>(param, constNode, false));
        return reshape;
    }
};

TEST_P(VPUXPropagateFQSubGraphTest_VPU3720, SW) {
    setPlatformVPU3720();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(VPUXPropagateFQSubGraphTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

const std::vector<std::vector<int64_t>> transposes = {
        {0, 3, 2, 1},
};

INSTANTIATE_TEST_CASE_P(smoke_PropagateFQSubGraph, VPUXPropagateFQSubGraphTest_VPU3720,
                        ::testing::ValuesIn(transposes));

}  // namespace
