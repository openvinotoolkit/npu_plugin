//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <base/behavior_test_utils.hpp>

#include "behavior/plugin/set_preprocess.hpp"
#include "common/functions.h"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"

using namespace LayerTestsUtils;

using namespace BehaviorTestsDefinitions;

namespace {
// Precision types
/*
    {InferenceEngine::Precision::I8,   InferenceEngine::Precision::U8,
    InferenceEngine::Precision::I16,  InferenceEngine::Precision::U16,
    InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32};
*/
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16,
                                                               InferenceEngine::Precision::FP32};

/* The Input and Output precisions conversion tests only cover U8 & FP32 currently
 Any other value will lead to asserts getting triggered */
const std::vector<InferenceEngine::Precision> inputPrecisions = {InferenceEngine::Precision::U8,
                                                                 InferenceEngine::Precision::FP32};
const std::vector<InferenceEngine::Precision> outputPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<std::map<std::string, std::string>> configs = {{}};

const std::vector<std::map<std::string, std::string>> autoConfigs = {
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU},
         {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
          ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU}}};

// Layout types:
/*
    {InferenceEngine::Layout::C,
    InferenceEngine::Layout::NC, InferenceEngine::Layout::CN, InferenceEngine::Layout::HW,
    InferenceEngine::Layout::CHW, InferenceEngine::Layout::HWC,
    InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC,
    InferenceEngine::Layout::NCDHW, InferenceEngine::Layout::NDHWC};
*/
const std::vector<InferenceEngine::Layout> netLayouts = {
        InferenceEngine::Layout::C,    InferenceEngine::Layout::NC,   InferenceEngine::Layout::CN,
        InferenceEngine::Layout::HW,   InferenceEngine::Layout::CHW,  InferenceEngine::Layout::HWC,
        InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC, InferenceEngine::Layout::NCDHW,
        InferenceEngine::Layout::NDHWC};

/*
The input layouts can only be set to NCHW or NHWC, due to the following
way in which the param is created to have 4 dimensions for this IR:
    unsigned int shape_size = 9, channels = 3, batch = 1, offset = 0;
    ngraph::PartialShape shape({batch, channels, shape_size, shape_size});
    auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
*/
const std::vector<InferenceEngine::Layout> inputLayouts = {InferenceEngine::Layout::NCHW,
                                                           InferenceEngine::Layout::NHWC};
// Output layout is similarly limited as the input layout
const std::vector<InferenceEngine::Layout> outputLayouts = {InferenceEngine::Layout::NCHW,
                                                            InferenceEngine::Layout::NHWC};
namespace InferRequestPreprocessTestName {
typedef std::tuple<InferenceEngine::Precision,         // Network precision
                   std::string,                        // Device name
                   std::map<std::string, std::string>  // Config
                   >
        BehaviorBasicParams;

static std::string getTestCaseName(testing::TestParamInfo<BehaviorBasicParams> obj) {
    using namespace ov::test::utils;

    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, targetDevice, configuration) = obj.param;
    std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
    targetDevice = LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    if (!configuration.empty()) {
        result << "config=" << configuration;
    }
    return result.str();
}
}  // namespace InferRequestPreprocessTestName

namespace InferRequestPreprocessConversionTestName {
typedef std::tuple<InferenceEngine::Precision,         // Network precision
                   InferenceEngine::Precision,         // Set input precision
                   InferenceEngine::Precision,         // Set output precision
                   InferenceEngine::Layout,            // Network layout - always NCHW
                   InferenceEngine::Layout,            // Set input layout
                   InferenceEngine::Layout,            // Set output layout
                   bool,                               // SetBlob or GetBlob for input blob
                   bool,                               // SetBlob or GetBlob for output blob
                   std::string,                        // Device name
                   std::map<std::string, std::string>  // Config
                   >
        PreprocessConversionParams;

static std::string getTestCaseName(testing::TestParamInfo<PreprocessConversionParams> obj) {
    InferenceEngine::Precision netPrecision, iPrecision, oPrecision;
    InferenceEngine::Layout netLayout, iLayout, oLayout;
    bool setInputBlob, setOutputBlob;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, iPrecision, oPrecision, netLayout, iLayout, oLayout, setInputBlob, setOutputBlob,
             targetDevice, configuration) = obj.param;
    std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
    targetDevice = LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);

    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "iPRC=" << iPrecision.name() << "_";
    result << "oPRC=" << oPrecision.name() << "_";
    result << "netLT=" << netLayout << "_";
    result << "iLT=" << iLayout << "_";
    result << "oLT=" << oLayout << "_";
    result << "setIBlob=" << setInputBlob << "_";
    result << "setOBlob=" << setOutputBlob << "_";
    result << "target_device=" << targetDevice;
    if (!configuration.empty()) {
        for (auto& configItem : configuration) {
            result << "configItem=" << configItem.first << "_" << configItem.second << "_";
        }
    }
    return result.str();
}
}  // namespace InferRequestPreprocessConversionTestName

INSTANTIATE_TEST_SUITE_P(smoke_precommit_BehaviorTestsPreprocess, InferRequestPreprocessTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestPreprocessTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_Auto_BehaviorTestsPreprocess, InferRequestPreprocessTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         InferRequestPreprocessTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_BehaviorTestsPreprocess, InferRequestPreprocessConversionTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(outputPrecisions), ::testing::ValuesIn(netLayouts),
                                            ::testing::ValuesIn(inputLayouts), ::testing::ValuesIn(outputLayouts),
                                            ::testing::Bool(), ::testing::Bool(),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestPreprocessConversionTestName::getTestCaseName);
}  // namespace
