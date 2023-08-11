// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/executable_network/exec_network_base.hpp"
#include "ie_plugin_config.hpp"

namespace BehaviorTestsDefinitions {

using VpuxExecutableNetworkBaseTest = ExecutableNetworkBaseTest;

TEST_P(VpuxExecutableNetworkBaseTest, VpuxCanExport) {
    const auto ts = CommonTestUtils::GetTimestamp();
    const std::string modelName = GetTestName().substr(0, CommonTestUtils::maxFileNameLength) + "_" + ts;
    auto execNet = ie->LoadNetwork(cnnNet, target_device, configuration);
    ASSERT_NO_THROW(execNet.Export(modelName + ".blob"));
    std::cout << "model name = " << modelName << std::endl;
    ASSERT_TRUE(CommonTestUtils::fileExists(modelName + ".blob"));
    CommonTestUtils::removeFile(modelName + ".blob");
}

}  // namespace BehaviorTestsDefinitions

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16, InferenceEngine::Precision::U8};

const std::vector<std::map<std::string, std::string>> configs = {{{"VPUX_CREATE_EXECUTOR", "0"}}};

const std::vector<std::map<std::string, std::string>> autoConfig = {
        {{MULTI_CONFIG_KEY(DEVICE_PRIORITIES), CommonTestUtils::DEVICE_KEEMBAY}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecutableNetworkBaseTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         ExecutableNetworkBaseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, VpuxExecutableNetworkBaseTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         VpuxExecutableNetworkBaseTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, ExecNetSetPrecision,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         ExecNetSetPrecision::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, ExecNetSetPrecision,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfig)),
                         ExecNetSetPrecision::getTestCaseName);

}  // namespace
