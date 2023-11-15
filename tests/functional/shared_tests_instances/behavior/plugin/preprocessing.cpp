//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/plugin/preprocessing.hpp"

#include "common/functions.h"
#include "common_test_utils/test_constants.hpp"
#include "vpu_ov1_layer_test.hpp"

using namespace LayerTestsUtils;

namespace BehaviorTestsDefinitions {

class VpuxPreprocessingPrecisionConvertTest :
        virtual public PreprocessingPrecisionConvertTest,
        virtual public LayerTestsUtils::VpuOv1LayerTestsCommon {
    void SetUp() override {
        PreprocessingPrecisionConvertTest::SetRefMode(LayerTestsUtils::RefMode::INTERPRETER);

        std::tie(PreprocessingPrecisionConvertTest::inPrc, channels, use_set_input,
                 PreprocessingPrecisionConvertTest::targetDevice, PreprocessingPrecisionConvertTest::configuration) =
                this->GetParam();
        PreprocessingPrecisionConvertTest::outPrc = PreprocessingPrecisionConvertTest::inPrc;

        auto make_ngraph = [&](bool with_extra_conv) {
            const auto inputShape = std::vector<size_t>{1, 3, 224, 224};
            const auto params = ngraph::builder::makeParams(ngraph::element::f32, {inputShape});
            const auto paramOuts = ngraph::helpers::convert2OutputVector(
                    ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
            const auto act_node = std::make_shared<ngraph::op::Relu>(paramOuts.at(0));
            const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(act_node)};
            return std::make_shared<ngraph::Function>(results, params, "ReLU_graph");
        };

        PreprocessingPrecisionConvertTest::function = make_ngraph(false);
        reference_function = make_ngraph(true);  // use extra ops to mimic the preprocessing
    }

public:
    static std::string getTestCaseName(testing::TestParamInfo<PreprocessingPrecisionConvertParams> obj) {
        return PreprocessingPrecisionConvertTest::getTestCaseName(obj) +
               "_targetPlatform=" + LayerTestsUtils::getTestsPlatformFromEnvironmentOr(CommonTestUtils::DEVICE_KEEMBAY);
    }
};

TEST_P(VpuxPreprocessingPrecisionConvertTest, PrecisionConvert) {
    PreprocessingPrecisionConvertTest::Run();
}

}  // namespace BehaviorTestsDefinitions

using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::I8, InferenceEngine::Precision::U8, InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32};

const std::vector<std::map<std::string, std::string>> configs = {{}};

const std::vector<std::map<std::string, std::string>> autoConfigs = {
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_KEEMBAY},
         {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
          CommonTestUtils::DEVICE_KEEMBAY + std::string(",") + CommonTestUtils::DEVICE_CPU}}};

INSTANTIATE_TEST_CASE_P(smoke_precommit_BehaviorTestsPreprocessingTests, VpuxPreprocessingPrecisionConvertTest,
                        ::testing::Combine(::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(1),  // Number of input tensor channels
                                           ::testing::Bool(),     // Use SetInput or GetBlob
                                           ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                           ::testing::ValuesIn(configs)),
                        VpuxPreprocessingPrecisionConvertTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_precommit_Auto_BehaviorTestsPreprocessingTests, VpuxPreprocessingPrecisionConvertTest,
                        ::testing::Combine(::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(1),  // Number of input tensor channels
                                           ::testing::Bool(),     // Use SetInput or GetBlob
                                           ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                           ::testing::ValuesIn(autoConfigs)),
                        VpuxPreprocessingPrecisionConvertTest::getTestCaseName);

}  // namespace
