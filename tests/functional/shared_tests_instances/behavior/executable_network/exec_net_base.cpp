// Copyright (C) 2018-2021 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "behavior/executable_network/exec_network_base.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "ie_plugin_config.hpp"

namespace BehaviorTestsDefinitions {

using VpuxExecutableNetworkBaseTest = ExecutableNetworkBaseTest;

TEST_P(VpuxExecutableNetworkBaseTest, VpuxCanExport) {
    const auto ts = ov::test::utils::GetTimestamp();
    const std::string modelName = GetTestName().substr(0, ov::test::utils::maxFileNameLength) + "_" + ts;
    auto execNet = ie->LoadNetwork(cnnNet, target_device, configuration);
    ASSERT_NO_THROW(execNet.Export(modelName + ".blob"));
    std::cout << "model name = " << modelName << std::endl;
    ASSERT_TRUE(ov::test::utils::fileExists(modelName + ".blob"));
    ov::test::utils::removeFile(modelName + ".blob");
}

}  // namespace BehaviorTestsDefinitions

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16, InferenceEngine::Precision::U8};

const std::vector<std::map<std::string, std::string>> configs = {{{"NPU_CREATE_EXECUTOR", "0"}}};

const std::vector<std::map<std::string, std::string>> autoConfig = {
        {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), ov::test::utils::DEVICE_NPU}},
};

namespace ExecutableNetworkBaseTestName {
static std::string getTestCaseName(testing::TestParamInfo<BehaviorTestsUtils::InferRequestParams> obj) {
    std::string target_device;
    std::map<std::string, std::string> configuration;
    std::tie(target_device, configuration) = obj.param;
    std::ostringstream result;
    std::replace(target_device.begin(), target_device.end(), ':', '.');
    result << "target_device=" << target_device << "_";
    result << "targetPlatform=" << LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU)
           << "_";
    if (!configuration.empty()) {
        using namespace ov::test::utils;
        result << "config=" << configuration;
    }
    return result.str();
}
}  // namespace ExecutableNetworkBaseTestName

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecutableNetworkBaseTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ExecutableNetworkBaseTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, VpuxExecutableNetworkBaseTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ExecutableNetworkBaseTestName::getTestCaseName);

namespace ExecNetSetPrecisionName {
static std::string getTestCaseName(testing::TestParamInfo<BehaviorTestsUtils::BehaviorBasicParams> obj) {
    using namespace ov::test::utils;

    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, targetDevice, configuration) = obj.param;
    std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
    std::ostringstream result;
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    result << "targetPlatform=" << LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU)
           << "_";
    if (!configuration.empty()) {
        result << "config=" << configuration;
    }
    return result.str();
}
}  // namespace ExecNetSetPrecisionName

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecNetSetPrecision,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ExecNetSetPrecisionName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, ExecNetSetPrecision,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfig)),
                         ExecNetSetPrecisionName::getTestCaseName);

}  // namespace
