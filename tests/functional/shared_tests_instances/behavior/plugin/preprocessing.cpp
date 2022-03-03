//
// Copyright 2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include "behavior/plugin/preprocessing.hpp"

#include "common_test_utils/test_constants.hpp"
#include "kmb_layer_test.hpp"
#include "common/functions.h"

using namespace LayerTestsUtils;

namespace BehaviorTestsDefinitions {

class VpuxPreprocessingPrecisionConvertTest : virtual public PreprocessingPrecisionConvertTest,
                                              virtual public LayerTestsUtils::KmbLayerTestsCommon{
    void SetUp() override {
        PreprocessingPrecisionConvertTest::SetRefMode(LayerTestsUtils::RefMode::INTERPRETER);

        std::tie(PreprocessingPrecisionConvertTest::inPrc, channels, use_set_input, PreprocessingPrecisionConvertTest::targetDevice, PreprocessingPrecisionConvertTest::configuration) = this->GetParam();
        PreprocessingPrecisionConvertTest::outPrc.front() = PreprocessingPrecisionConvertTest::inPrc;

        auto make_ngraph = [&](bool with_extra_conv) {
            const auto inputShape = std::vector<size_t>{1, 3, 224, 224};
            const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
            const auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
            const auto act_node = std::make_shared<ngraph::op::Relu>(paramOuts.at(0));
            const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(act_node)};
            return std::make_shared<ngraph::Function>(results, params, "ReLU_graph");
        };

        PreprocessingPrecisionConvertTest::function            = make_ngraph(false);
        reference_function  = make_ngraph(true);  //use extra ops to mimic the preprocessing
    }
};


TEST_P(VpuxPreprocessingPrecisionConvertTest, PrecisionConvert) {
    PreprocessingPrecisionConvertTest::Run();
}

}

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {InferenceEngine::Precision::U8,
                                                                 InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{}};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTestsPreprocessingTestsViaSetInput, VpuxPreprocessingPrecisionConvertTest,
                        ::testing::Combine(::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(1),  // Number of input tensor channels
                                           ::testing::Values(true),           // Use SetInput
                                           ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                           ::testing::ValuesIn(configs)),
                        VpuxPreprocessingPrecisionConvertTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
        smoke_BehaviorTestsPreprocessingTestsViaGetBlob, VpuxPreprocessingPrecisionConvertTest,
        ::testing::Combine(
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(1),  // Number of input tensor channels
                ::testing::Values(false),  // use GetBlob
                ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY), ::testing::ValuesIn(configs)),
        VpuxPreprocessingPrecisionConvertTest::getTestCaseName);
}  // namespace
