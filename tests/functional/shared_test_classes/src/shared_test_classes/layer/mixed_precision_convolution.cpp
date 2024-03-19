//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "shared_test_classes/layer/mixed_precision_convolution.hpp"

using ngraph::helpers::QuantizationGranularity;

namespace LayerTestsDefinitions {

std::string MixedPrecisionConvLayerTest::getTestCaseName(
        const testing::TestParamInfo<mixedPrecisionConvLayerTestParamsSet>& obj) {
    mixedPrecisionConvSpecificParams mixedPrecisionConvParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    ov::test::TargetDevice device;
    std::tie(mixedPrecisionConvParams, netPrecision, inputShapes, device) = obj.param;
    ngraph::op::PadType padType = ngraph::op::PadType::AUTO;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    size_t quantLevels;
    QuantizationGranularity quantGranularity;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, quantLevels, quantGranularity) =
            mixedPrecisionConvParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "Levels=" << quantLevels << "_";
    result << "QG=" << quantGranularity << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "device=" << device;
    return result.str();
}

void MixedPrecisionConvLayerTest::SetUp() {
    abs_threshold = 1.0f;

    mixedPrecisionConvSpecificParams mixedPrecisionConvParams;
    std::vector<size_t> inputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(mixedPrecisionConvParams, netPrecision, inputShape, targetDevice) = this->GetParam();
    ngraph::op::PadType padType = ngraph::op::PadType::AUTO;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    size_t quantLevels;
    size_t quantGranularity;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, quantLevels, quantGranularity) =
            mixedPrecisionConvParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto paramOuts =
            ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));

    std::vector<size_t> weightsShapes = {convOutChannels, inputShape[1], kernel[0], kernel[1]};

    std::mt19937 intMersenneEngine{0};
    std::normal_distribution<double> intDist{-127.0, 127.0};
    auto intGen = [&intDist, &intMersenneEngine]() {
        return (int8_t)std::round(intDist(intMersenneEngine));
    };

    std::vector<int8_t> weightsData(ngraph::shape_size(weightsShapes));
    std::generate(weightsData.begin(), weightsData.end(), intGen);
    const auto weightsConst =
            std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i8, weightsShapes, weightsData);

    const auto weightsConvert = std::make_shared<ngraph::op::v0::Convert>(weightsConst, ngraph::element::Type_t::f16);

    std::vector<double> multiplyData(weightsShapes[0], 0.078740157480314959);
    const auto multiplyConst = std::make_shared<ngraph::op::Constant>(
            ngraph::element::Type_t::f16, ngraph::Shape({weightsShapes[0], 1, 1, 1}), multiplyData);
    const auto weightsMultiply = std::make_shared<ngraph::op::v1::Multiply>(weightsConvert, multiplyConst);

    auto conv = std::make_shared<ngraph::op::v1::Convolution>(paramOuts[0], weightsMultiply, stride, padBegin, padEnd,
                                                              dilation, padType);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(conv)};
    function = std::make_shared<ngraph::Function>(results, params, "MixedPrecisionConvolution");
}
}  // namespace LayerTestsDefinitions
