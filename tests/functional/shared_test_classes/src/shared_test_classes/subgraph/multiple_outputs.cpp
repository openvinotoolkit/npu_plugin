// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/multiple_outputs.hpp"
#include "ngraph_functions/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string MultioutputTest::getTestCaseName(testing::TestParamInfo<multiOutputTestParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    size_t outputChannels;
    convParams convolutionParams;
    std::vector<size_t> inputShape;
    std::vector<size_t> kernelShape;
    size_t stride;
    std::tie(netPrecision, targetDevice, configuration, convolutionParams, outputChannels) = obj.param;
    std::tie(inputShape, kernelShape, stride) = convolutionParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "KS=" << CommonTestUtils::vec2str(kernelShape) << "_";
    result << "S=" << stride << "_";
    result << "OC=" << outputChannels << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

InferenceEngine::Blob::Ptr MultioutputTest::GenerateInput(const InferenceEngine::InputInfo& info) const {
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
    blob->allocate();
    auto precision = info.getPrecision();

    auto* rawBlobDataPtr = blob->buffer().as<float*>();
    for (size_t i = 0; i < blob->size(); i++) {
        float value = i % 16;
        if (typeid(precision) == typeid(typename InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP16>::value_type)) {
            rawBlobDataPtr[i] = ngraph::float16(value).to_bits();
        } else {
            rawBlobDataPtr[i] = value;
        }
    }
    return blob;
}

void MultioutputTest::SetUp() 
{
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    convParams convolutionParams;
    size_t outputChannels;
    std::tie(netPrecision, targetDevice, tempConfig, convolutionParams, outputChannels) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    std::vector<size_t> inputShape;
    std::vector<size_t> kernelShape;
    size_t stride;
    std::tie(inputShape, kernelShape, stride) = convolutionParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    
    // input
    auto params = ngraph::builder::makeParams(ngPrc, { inputShape });
    // conv 1
    auto conv1Weights = CommonTestUtils::generate_float_numbers(outputChannels * inputShape[1] * kernelShape[0] * kernelShape[1], -0.2f, 0.2f);
    auto conv1 = ngraph::builder::makeConvolution(params[0], ngPrc, { kernelShape[0], kernelShape[1] }, { stride, stride }, { 1, 1 },
        { 1, 1 }, { 1, 1 }, ngraph::op::PadType::VALID, outputChannels, false, conv1Weights);
    // conv 2
    std::vector<size_t> conv2InputShape = {1, outputChannels, inputShape[2], inputShape[3]};
    auto conv2Weights = CommonTestUtils::generate_float_numbers(outputChannels * conv2InputShape[1] * kernelShape[0] * kernelShape[1], -0.2f, 0.2f);
    auto conv2 = ngraph::builder::makeConvolution(conv1, ngPrc, { kernelShape[0], kernelShape[1] }, { stride, stride }, { 0, 0 },
        { 0, 0 }, { 1, 1 }, ngraph::op::PadType::VALID, outputChannels, true, conv2Weights);    
    // max pool
    auto pool = ngraph::builder::makePooling(conv2, {1, 1}, {0, 0}, {0, 0}, {2, 2}, ngraph::op::RoundingType::FLOOR,
        ngraph::op::PadType::VALID, false, ngraph::helpers::PoolingTypes::MAX);

    ngraph::ResultVector results{std::make_shared<ngraph::op::Result>(conv1),
                                 std::make_shared<ngraph::op::Result>(pool) };
    function = std::make_shared<ngraph::Function>(results, params, "MultioutputTest");
}

}  // namespace SubgraphTestsDefinitions
