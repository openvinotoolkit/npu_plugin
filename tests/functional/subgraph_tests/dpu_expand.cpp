//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "subgraph_tests/nce_tasks.hpp"
#include "vpu_ov1_layer_test.hpp"

#include <ngraph_functions/builders.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

namespace {

struct ShapeWithNumFilters {
    ShapeWithNumFilters(const InferenceEngine::SizeVector& shape, const size_t filters, const bool bypassQuant)
            : _inShape(shape), _outFilters(filters), _bypassQuant(bypassQuant) {
    }
    ShapeWithNumFilters(const ShapeWithNumFilters& other)
            : _inShape(other._inShape), _outFilters(other._outFilters), _bypassQuant(other._bypassQuant) {
    }
    ShapeWithNumFilters(const ShapeWithNumFilters&& other) = delete;
    ShapeWithNumFilters& operator=(const ShapeWithNumFilters& other) {
        _inShape = other._inShape;
        _outFilters = other._outFilters;
        _bypassQuant = other._bypassQuant;
        return *this;
    }
    ShapeWithNumFilters& operator=(const ShapeWithNumFilters&& other) = delete;
    ~ShapeWithNumFilters() = default;
    InferenceEngine::SizeVector _inShape;
    size_t _outFilters;
    bool _bypassQuant;
};

class VPUXConv2dWithExpandTest_VPU3720 :
        public LayerTestsUtils::VpuOv1LayerTestsCommon,
        public testing::WithParamInterface<ShapeWithNumFilters> {
    void ConfigureNetwork() override {
        cnnNetwork.getInputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getOutputsInfo().begin()->second->setPrecision(InferenceEngine::Precision::FP16);
        cnnNetwork.getInputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
        cnnNetwork.getOutputsInfo().begin()->second->setLayout(InferenceEngine::Layout::NHWC);
    }
    ov::Output<ov::Node> quantizeValue(const ov::Output<ov::Node>& param, const bool bypassQuant,
                                       const std::array<float, 4>& quantRange, const size_t quantLevels) const {
        if (bypassQuant) {
            return param;
        }
        const auto fqOp = NCETasksHelpers::quantize(param, quantRange, quantLevels);
        return fqOp->output(0);
    }
    ov::Output<ov::Node> composeFloatWeights(const ngraph::Shape& weightsShape) const {
        const size_t totalSize =
                std::accumulate(weightsShape.begin(), weightsShape.end(), 1, std::multiplies<size_t>());
        std::vector<float> weights(totalSize, 0.f);
        for (size_t i = 0; i < weights.size(); i++) {
            weights.at(i) = std::cos(i * 3.14 / 6);
        }
        const auto weightsConst = ngraph::opset8::Constant::create(ngraph::element::Type_t::f32, weightsShape, weights);
        return weightsConst->output(0);
    }
    ov::Output<ov::Node> composeQuantWeights(const ngraph::Shape& weightsShape) const {
        const size_t strideKX = 1;
        const size_t strideKY = strideKX * weightsShape[3];
        const size_t strideIC = strideKY * weightsShape[2];
        const size_t strideOC = strideIC * weightsShape[1];
        const size_t totalSize =
                std::accumulate(weightsShape.begin(), weightsShape.end(), 1, std::multiplies<size_t>());
        std::vector<float> weights(totalSize, 0.f);
        for (size_t oc = 0; oc < weightsShape[0]; oc++) {
            const size_t ic = oc % weightsShape[1];
            for (size_t ky = 0; ky < weightsShape[2]; ky++) {
                for (size_t kx = 0; kx < weightsShape[3]; kx++) {
                    const size_t idx = kx * strideKX + ky * strideKY + ic * strideIC + oc * strideOC;
                    weights.at(idx) = 127.f;
                }
            }
        }
        const auto weightsConst = ngraph::opset8::Constant::create(ngraph::element::Type_t::f32, weightsShape, weights);
        const auto quantWeightsRange = std::array<float, 4>{-127.f, 127.f, -1.f, 1.f};
        const auto convWeights = quantizeValue(weightsConst->output(0), false, quantWeightsRange, 255);

        return convWeights;
    }
    ov::Output<ov::Node> composeWeights(const bool bypassQuant, const size_t outputChannels,
                                        const size_t inputChannels) const {
        const size_t kernelX = 1;
        const size_t kernelY = 1;
        const auto weightsShape = ngraph::Shape{outputChannels, inputChannels, kernelY, kernelX};
        if (bypassQuant) {
            return composeFloatWeights(weightsShape);
        }
        return composeQuantWeights(weightsShape);
    }
    void SetUp() override {
        targetDevice = LayerTestsUtils::testPlatformTargetDevice();
        const auto testParameters = GetParam();
        const InferenceEngine::SizeVector& inputShape = testParameters._inShape;
        const bool& bypassQuant = testParameters._bypassQuant;
        const size_t& inputChannels = inputShape.at(1);
        const size_t& outputChannels = testParameters._outFilters;

        const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
        const auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        const auto quantInputRange = std::array<float, 4>{-128.f, 127.f, -128.f, 127.f};
        const auto convInput = quantizeValue(paramOuts.at(0), bypassQuant, quantInputRange, 256);
        const auto convWeights = composeWeights(bypassQuant, outputChannels, inputChannels);

        const auto conv = std::make_shared<ngraph::op::v1::Convolution>(
                convInput, convWeights, ngraph::Strides(std::vector<size_t>{1, 1}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}),
                ngraph::CoordinateDiff(std::vector<ptrdiff_t>{0, 0}), ngraph::Strides(std::vector<size_t>{1, 1}));

        const auto quantOutputRange = std::array<float, 4>{-128.f, 127.f, -128.f, 127.f};
        const auto convOutput = quantizeValue(conv->output(0), bypassQuant, quantOutputRange, 256);

        const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(convOutput)};

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
        ShapeWithNumFilters({1, 3, 112, 224}, 16, true),  ShapeWithNumFilters({1, 24, 64, 112}, 32, true),
        ShapeWithNumFilters({1, 1, 19, 80}, 16, true),    ShapeWithNumFilters({1, 3, 112, 224}, 16, false),
        ShapeWithNumFilters({1, 24, 64, 112}, 32, false), ShapeWithNumFilters({1, 1, 19, 80}, 16, false),
};

INSTANTIATE_TEST_SUITE_P(conv2d_with_expand, VPUXConv2dWithExpandTest_VPU3720, ::testing::ValuesIn(shapes));

}  // namespace
