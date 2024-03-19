// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "subgraph_tests/quantized_group_convolution.hpp"
#include "common_test_utils/test_constants.hpp"
#include "vpux_private_properties.hpp"

#include <vector>

#include "vpu_ov1_layer_test.hpp"

namespace SubgraphTestsDefinitions {

// MLIR detects pattern quant.dcast -> op -> quant.qcast and converts it into single quantized Op
//
//       [input]
//          |
//     (dequantize)
//          |
//        (conv) --- (dequantize) -- [filter]
//          |
//       [output]
//          |
//      (quantize)
//

class QuantGroupConvLayerTest_NPU3700 :
        // API 1.0 usage in OV
        // [Tracking number: E#97126]
        public QuantGroupConvLayerTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SetUp() override {
        threshold = 0.5f;

        quantGroupConvSpecificParams groupConvParams;
        std::vector<size_t> inputShape;
        auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
        std::tie(groupConvParams, netPrecision, inputShape, targetDevice) = this->GetParam();
        ngraph::op::PadType padType = ngraph::op::PadType::AUTO;
        InferenceEngine::SizeVector kernel, stride, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t convOutChannels, numGroups;
        size_t quantLevels;
        size_t quantGranularity;
        bool quantizeWeights;
        std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, quantLevels, quantGranularity,
                 quantizeWeights) = groupConvParams;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
        auto paramOuts =
                ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        std::vector<size_t> dataFqConstShapes(inputShape.size(), 1);
        if (quantGranularity == ngraph::helpers::Perchannel)
            dataFqConstShapes[1] = inputShape[1];
        auto dataFq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngPrc, quantLevels, dataFqConstShapes, {0}, {255},
                                                        {0}, {255});

        std::vector<size_t> weightsShapes = {convOutChannels, inputShape[1]};
        if (weightsShapes[0] % numGroups || weightsShapes[1] % numGroups)
            throw std::runtime_error("incorrect shape for QuantGroupConvolution");
        weightsShapes[0] /= numGroups;
        weightsShapes[1] /= numGroups;
        weightsShapes.insert(weightsShapes.begin(), numGroups);
        weightsShapes.insert(weightsShapes.end(), kernel.begin(), kernel.end());

        std::vector<float> weightsData;
        std::shared_ptr<ngraph::Node> weights;
        if (quantizeWeights) {
            std::vector<size_t> fqWeightsShapes{convOutChannels, inputShape[1] / numGroups};
            fqWeightsShapes.insert(fqWeightsShapes.end(), kernel.begin(), kernel.end());

            std::vector<size_t> weightsFqConstShapes(inputShape.size(), 1);
            if (quantGranularity == ngraph::helpers::Perchannel)
                weightsFqConstShapes[0] = fqWeightsShapes[0];

            auto weightsNode = ngraph::builder::makeConstant(ngPrc, fqWeightsShapes, weightsData, weightsData.empty());
            auto fqNode = ngraph::builder::makeFakeQuantize(weightsNode, ngPrc, quantLevels, weightsFqConstShapes, {0},
                                                            {255}, {0}, {255});

            auto constNode = std::make_shared<ov::op::v0::Constant>(ngraph::element::Type_t::i64,
                                                                    ngraph::Shape{weightsShapes.size()}, weightsShapes);
            weights = std::dynamic_pointer_cast<ov::op::v1::Reshape>(
                    std::make_shared<ov::op::v1::Reshape>(fqNode, constNode, false));
        } else {
            auto weightsNode = ngraph::builder::makeConstant(ngPrc, weightsShapes, weightsData, weightsData.empty());
            weights = weightsNode;
        }

        auto groupConv = std::dynamic_pointer_cast<ov::op::v1::GroupConvolution>(ngraph::builder::makeGroupConvolution(
                dataFq, weights, ngPrc, stride, padBegin, padEnd, dilation, padType));

        const auto outFq = ngraph::builder::makeFakeQuantize(groupConv, ngPrc, quantLevels, {}, {0}, {255}, {0}, {255});

        ngraph::ResultVector results{std::make_shared<ov::op::v0::Result>(outFq)};
        function = std::make_shared<ngraph::Function>(results, params, "QuantGroupConvolution");
    }
};

TEST_P(QuantGroupConvLayerTest_NPU3700, SW) {
    setPlatformVPU3700();
    setReferenceSoftwareModeMLIR();
    Run();
}

TEST_P(QuantGroupConvLayerTest_NPU3700, HW) {
    setPlatformVPU3700();
    setDefaultHardwareModeMLIR();
    Run();
}

}  // namespace SubgraphTestsDefinitions

using namespace SubgraphTestsDefinitions;
using namespace ngraph::helpers;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16};

const std::vector<size_t> numOutChannels = {3, 24, 48};
const std::vector<size_t> numGroups = {3};

const std::vector<size_t> levels = {256};
const std::vector<QuantizationGranularity> granularity = {Pertensor, Perchannel};
const std::vector<bool> quantizeWeights2D = {true};

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<size_t>> inputShapes2D = {{1, 3, 10, 10}, {1, 24, 10, 10}};
const std::vector<std::vector<size_t>> kernels2D = {{1, 1}, {3, 3}};
const std::vector<std::vector<size_t>> strides2D = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{0, 0}};
const std::vector<std::vector<size_t>> dilations2D = {{1, 1}};

const auto quantGroupConv2DParams = ::testing::Combine(
        ::testing::ValuesIn(kernels2D), ::testing::ValuesIn(strides2D), ::testing::ValuesIn(padBegins2D),
        ::testing::ValuesIn(padEnds2D), ::testing::ValuesIn(dilations2D), ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups), ::testing::ValuesIn(levels), ::testing::ValuesIn(granularity),
        ::testing::ValuesIn(quantizeWeights2D));

INSTANTIATE_TEST_SUITE_P(smoke_QuantGroupConv2D, QuantGroupConvLayerTest_NPU3700,
                         ::testing::Combine(quantGroupConv2DParams, ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(inputShapes2D),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         QuantGroupConvLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<std::vector<size_t>> inputShapes3D = {{1, 3, 5, 5, 5}, {1, 24, 5, 5, 5}};
const std::vector<std::vector<size_t>> kernels3D = {{3, 3, 3}};
const std::vector<std::vector<size_t>> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{0, 0, 0}};
const std::vector<std::vector<size_t>> dilations3D = {{1, 1, 1}};
const std::vector<bool> quantizeWeights3D = {true};

const auto quantGroupConv3DParams = ::testing::Combine(
        ::testing::ValuesIn(kernels3D), ::testing::ValuesIn(strides3D), ::testing::ValuesIn(padBegins3D),
        ::testing::ValuesIn(padEnds3D), ::testing::ValuesIn(dilations3D), ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups), ::testing::ValuesIn(levels), ::testing::ValuesIn(granularity),
        ::testing::ValuesIn(quantizeWeights3D));

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_QuantGroupConv3D, QuantGroupConvLayerTest_NPU3700,
                         ::testing::Combine(quantGroupConv3DParams, ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(inputShapes3D),
                                            ::testing::Values(LayerTestsUtils::testPlatformTargetDevice())),
                         QuantGroupConvLayerTest::getTestCaseName);

}  // namespace
