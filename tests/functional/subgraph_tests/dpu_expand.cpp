// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

struct ShapeWithNumFilters {
    ShapeWithNumFilters(const InferenceEngine::SizeVector& shape, const size_t filters)
            : _inShape(shape), _outFilters(filters) {
    }
    ShapeWithNumFilters(const ShapeWithNumFilters& other): _inShape(other._inShape), _outFilters(other._outFilters) {
    }
    ShapeWithNumFilters(const ShapeWithNumFilters&& other) = delete;
    ShapeWithNumFilters& operator=(const ShapeWithNumFilters& other) {
        _inShape = other._inShape;
        _outFilters = other._outFilters;
        return *this;
    }
    ShapeWithNumFilters& operator=(const ShapeWithNumFilters&& other) = delete;
    ~ShapeWithNumFilters() = default;
    InferenceEngine::SizeVector _inShape;
    size_t _outFilters;
};

class VPUXConv2dWithExpandTest_VPU3720 :
        public LayerTestsUtils::KmbLayerTestsCommon,
        public testing::WithParamInterface<ShapeWithNumFilters> {
    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
        cnnNetwork.getOutputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
    }
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice;
        const size_t kernelX = 1;
        const size_t kernelY = 1;
        const auto testParameters = GetParam();
        const InferenceEngine::SizeVector& inputShape = testParameters._inShape;
        const size_t& inputChannels = inputShape.at(1);
        const size_t& outputChannels = testParameters._outFilters;

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        std::vector<float> weights(inputChannels * outputChannels * kernelX * kernelY);
        for (size_t i = 0; i < weights.size(); i++) {
            weights.at(i) = std::cos(i * 3.14 / 6);
        }
        const auto weightsShape = ngraph::Shape{outputChannels, inputChannels, kernelY, kernelX};
        const auto weightsConst = ngraph::opset8::Constant::create(ngraph::element::Type_t::f32, weightsShape, weights);

        const auto conv = std::make_shared<ngraph::op::v1::Convolution>(
                paramOuts.at(0), weightsConst->output(0), ngraph::Strides(std::vector<size_t>{1, 1}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ngraph::Strides(std::vector<size_t>{1, 1}));

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(conv)};

        function = std::make_shared<ngraph::Function>(results, params, "VPUXConv2dWithExpandTest");

        threshold = 0.5f;
    }
};

TEST_P(VPUXConv2dWithExpandTest_VPU3720, HW) {
    setPlatformVPU3720();
    setDefaultHardwareModeMLIR();
    Run();
}

const std::vector<ShapeWithNumFilters> shapes = {
        ShapeWithNumFilters({1, 3, 112, 224}, 16),
        ShapeWithNumFilters({1, 24, 64, 112}, 32),
        ShapeWithNumFilters({1, 1, 19, 80}, 16),
};

INSTANTIATE_TEST_SUITE_P(conv2d_with_expand, VPUXConv2dWithExpandTest_VPU3720, ::testing::ValuesIn(shapes));

}  // namespace
