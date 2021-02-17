// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/multiple_output_test.hpp"
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

void MultioutputTest::SetUp() {
    auto generateWeights = [](std::size_t out_channels, std::size_t kernel_size) {
        std::vector<float> res;
        for (int i = 0; i < out_channels; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                j == 0 ? res.emplace_back(1.0f) : res.emplace_back(0.0f);
            }
        }

        return res;
    };

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
    auto params = ngraph::builder::makeParams(ngPrc, { inputShape });

    auto conv0 = ngraph::builder::makeConvolution(params[0], ngPrc, { kernelShape[0], kernelShape[1] }, { stride, stride }, { 0, 0 },
        { 0, 0 }, { 1, 1 }, ngraph::op::PadType::VALID, outputChannels, true,
        generateWeights(outputChannels, kernelShape[1]));

    ngraph::ResultVector results{ std::make_shared<ngraph::op::Result>(conv0) };
    function = std::make_shared<ngraph::Function>(results, params, "InputConvTest");
}

}  // namespace SubgraphTestsDefinitions

