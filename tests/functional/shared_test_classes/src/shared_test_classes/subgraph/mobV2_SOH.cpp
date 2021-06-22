#include "shared_test_classes/subgraph/mobV2_SOH.hpp"
#include <transformations/serialize.hpp>
#include "ngraph_functions/builders.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;
using namespace LayerTestsDefinitions;

std::string mobilenetV2SlicedTest::getTestCaseName(const testing::TestParamInfo<mobilenetV2SlicedParameters> &obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, targetDevice, configuration) = obj.param;

    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void mobilenetV2SlicedTest::SetUp() {

    /* creates subgraph
           input
             |
          groupConv
             |
            Add1
             |
           Clamp
             |
           Conv
             |
            Add2
             |
           output
    */
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    std::tie(netPrecision, targetDevice, tempConfig) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    
    // input
    std::vector<size_t> inputShape = { 1, 144, 56, 56 };
    const auto params = ngraph::builder::makeParams(ngPrc, { inputShape });
    // GroupConv
    const auto groupConvWeights = CommonTestUtils::generate_float_numbers(144 * 3 * 3, -0.2f, 0.2f);
    const auto groupConv = ngraph::builder::makeGroupConvolution(params[0], ngPrc, { 3 ,3 },  { 2, 2 } , { 1, 1 }, 
                        { 1, 1 }, { 1, 1 }, ngraph::op::PadType::EXPLICIT, 144, 144, false, groupConvWeights);
    //Add1
    const std::vector<float> bias = CommonTestUtils::generate_float_numbers(144, -5.f, 5.f);
    const auto biasNode = ngraph::builder::makeConstant<float>(ngPrc, {1, 144, 1, 1}, bias, false);
    const auto add1 = std::make_shared<ngraph::opset1::Add>(groupConv, biasNode);
    //Clamp
    const auto clamp = std::make_shared<ngraph::op::v0::Clamp>(add1, 0.0f, 6.0f);
    // conv
    std::vector<size_t> convInputShape = { 1, 144, 28, 28 };
    const auto convWeights = CommonTestUtils::generate_float_numbers(32 * convInputShape[1] * 1 * 1, -0.2f, 0.2f);
    const auto conv = ngraph::builder::makeConvolution(clamp, ngPrc, { 1, 1 },  { 1, 1 } , { 0, 0 },
                    { 0, 0 }, { 1, 1 }, ngraph::op::PadType::EXPLICIT, 32, false, convWeights);    

    //Add2
    const std::vector<float> bias1 = CommonTestUtils::generate_float_numbers(32, -5.f, 5.f);
    const auto biasNode1 = ngraph::builder::makeConstant<float>(ngPrc, {1, 32, 1, 1}, bias1, false);
    const auto add2 = std::make_shared<ngraph::opset1::Add>(conv, biasNode1);
    
    //result
    ngraph::ResultVector results{std::make_shared<ngraph::op::Result>(add2)};
    
    function = std::make_shared<ngraph::Function>(results, params, "MobilenetV2SlicedTest");
}
