#include "conv_act_base.hpp"
#include <transformations/serialize.hpp>

using namespace SubgraphTestsDefinitions;
using namespace LayerTestsDefinitions;

std::string ConvActTest::getTestCaseName(const testing::TestParamInfo<convActTestParamsSet> &obj) {
    LayerTestsDefinitions::activationParams aParams;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(aParams, kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = obj.param;

    std::ostringstream result;
    auto accPartName = LayerTestsDefinitions::ActivationLayerTest::getTestCaseName({aParams, 0});

    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << accPartName;
    return result.str();
}

void ConvActTest::buildFloatFunction() {
    auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;

    LayerTestsDefinitions::activationParams aParams;
    std::tie(aParams, kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = GetParam();

    std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
    std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationDecl;
    std::tie(activationDecl, netPrecision, inPrc, outPrc, inLayout, outLayout, shapes, targetDevice) = aParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {shapes.first});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    std::vector<float> filter_weights;
    auto filter_size = std::accumulate(std::begin(kernel), std::end(kernel), 1, std::multiplies<size_t>());
    filter_weights = CommonTestUtils::generate_float_numbers(convOutChannels * shapes.first[1] * filter_size,
                                                             -0.5f, 0.5f);
    auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(
            ngraph::builder::makeConvolution(paramOuts[0], ngPrc, kernel, stride, padBegin,
                                             padEnd, dilation, padType, convOutChannels, false, filter_weights));

    ngraph::ResultVector results{};

    ngraph::helpers::ActivationTypes activationType;
    activationType = activationDecl.first;
    auto constantsValue = activationDecl.second;
    auto activation = ngraph::builder::makeActivation(conv, ngPrc, activationType, shapes.second, constantsValue);
    results.push_back(std::make_shared<ngraph::opset1::Result>(activation));

    function = std::make_shared<ngraph::Function>(results, params, "convolution");

    /*
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>("Test.xml", "Test.bin");
    manager.run_passes(function); */
}

void ConvActTest::buildFQFunction() {
    auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;

    LayerTestsDefinitions::activationParams aParams;
    std::tie(aParams, kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = GetParam();

    std::pair<std::vector<size_t>, std::vector<size_t>> shapes;
    std::pair<ngraph::helpers::ActivationTypes, std::vector<float>> activationDecl;
    std::tie(activationDecl, netPrecision, inPrc, outPrc, inLayout, outLayout, shapes, targetDevice) = aParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    /// building conv+activation+FQs subgraph

    const InferenceEngine::SizeVector inputShape = shapes.first;// {1, 3, 62, 62};

    auto filter_size = std::accumulate(std::begin(kernel), std::end(kernel), 1, std::multiplies<size_t>());

    const InferenceEngine::SizeVector weightsShape {convOutChannels * filter_size, shapes.first[1], 1, 1};

    const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
    const auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    /// building data FQ
    const size_t dataLevels = 256;
    const std::vector<float> dataLow = {0.f};
    const std::vector<float> dataHigh = {254.125f};
    const auto dataFq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngraph::element::f32, dataLevels, {}, dataLow, dataHigh, dataLow, dataHigh);

    /// building weights FQ - through convert layer
    const auto weightsU8 = ngraph::builder::makeConstant<uint8_t>(ngraph::element::u8, weightsShape, {}, true, 255, 1);
    const auto weightsFP32 = std::make_shared<ngraph::opset2::Convert>(weightsU8, ngraph::element::f32);

    const size_t weightsLevels = 255;

    const auto weightsInLow = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1}, {0.0f}, false);
    const auto weightsInHigh = ngraph::builder::makeConstant<float>(ngraph::element::f32, {1}, {254.0f}, false);

    std::vector<float> perChannelLow(weightsShape[0]);
    std::vector<float> perChannelHigh(weightsShape[0]);

    for (size_t i = 0; i < weightsShape[0]; ++i) {
        perChannelLow[i] = -1.0f;
        perChannelHigh[i] = 1.0f;
    }

    const auto weightsOutLow = ngraph::builder::makeConstant<float>(ngraph::element::f32, {weightsShape[0], 1, 1, 1}, perChannelLow, false);
    const auto weightsOutHigh = ngraph::builder::makeConstant<float>(ngraph::element::f32, {weightsShape[0], 1, 1, 1}, perChannelHigh, false);
    const auto weightsFq = std::make_shared<ngraph::opset2::FakeQuantize>(weightsFP32, weightsInLow, weightsInHigh, weightsOutLow, weightsOutHigh, weightsLevels);

    /// building convolution
    const ngraph::Strides strides = {1, 1};
    const ngraph::CoordinateDiff pads_begin = {0, 0};
    const ngraph::CoordinateDiff pads_end = {0, 0};
    const ngraph::Strides dilations = {1, 1};
    const auto conv = std::make_shared<ngraph::opset2::Convolution>(dataFq, weightsFq, strides, pads_begin, pads_end, dilations);

    /// building activation
    ngraph::helpers::ActivationTypes activationType;
    activationType = activationDecl.first;
    auto constantsValue = activationDecl.second;
    auto activation = ngraph::builder::makeActivation(conv, ngPrc, activationType, shapes.second, constantsValue);


    /// activation FQ
    const std::vector<float> outDataLow = {-14.0f};
    const std::vector<float> outDataHigh = {14.0f};
    const auto activationaFq = ngraph::builder::makeFakeQuantize(activation, ngraph::element::f32, dataLevels, {}, outDataLow, outDataHigh, outDataLow, outDataHigh);


    const ngraph::ResultVector results{
            std::make_shared<ngraph::opset1::Result>(activationaFq)
    };
    function = std::make_shared<ngraph::Function>(results, params, "KmbQuantizedConvAcc");

    threshold = 0.4f;


    /*ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::Serialize>("Test.xml", "Test.bin");
    manager.run_passes(function);*/
}

void ConvActTest::SetUp() {
    auto accParams = std::get<0>(GetParam());
    InferenceEngine::Precision netPrecision= std::get<1>(accParams);

    switch (netPrecision.getPrecVal()) {
    case InferenceEngine::Precision::FP32 :
    case InferenceEngine::Precision::FP16 :
        buildFloatFunction();
        return;
    case InferenceEngine::Precision::U8 :
        buildFQFunction();
        return;
    default:
        FAIL() << "unsupported network precision for test case: " << netPrecision ;
    }
}
