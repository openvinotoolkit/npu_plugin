// Copyright (C) 2018-2021 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "behavior/infer_request/io_blob.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "ie_plugin_config.hpp"
#include "overload/infer_request/io_blob.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> configs = {{}};

const std::vector<std::map<std::string, std::string>> autoconfigs = {
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, ov::test::utils::DEVICE_NPU}},
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
          ov::test::utils::DEVICE_NPU + std::string(",") + ov::test::utils::DEVICE_CPU}}};

namespace InferRequestIOBBlobSetLayoutTestName {
static std::string getTestCaseName(testing::TestParamInfo<InferRequestIOBBlobSetLayoutParams> obj) {
    using namespace ov::test::utils;
    InferenceEngine::Layout layout;
    std::string target_device;
    std::map<std::string, std::string> configuration;
    std::tie(layout, target_device, configuration) = obj.param;
    target_device = LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);
    std::ostringstream result;
    result << "layout=" << layout << "_";
    result << "target_device=" << target_device << "_";
    if (!configuration.empty()) {
        result << "config=" << configuration;
    }
    return result.str();
}
}  // namespace InferRequestIOBBlobSetLayoutTestName

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestIOBBlobTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestParamsMapTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestIOBBlobTest,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoconfigs)),
                         InferRequestParamsMapTestName::getTestCaseName);

// Ticket: E-80555
INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestIOBBlobTestVpux,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestParamsMapTestName::getTestCaseName);

// Ticket: E-80555
INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestIOBBlobTestVpux,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoconfigs)),
                         InferRequestParamsMapTestName::getTestCaseName);

std::vector<InferenceEngine::Layout> layouts = {
        InferenceEngine::Layout::ANY,    InferenceEngine::Layout::NCHW,    InferenceEngine::Layout::NHWC,
        InferenceEngine::Layout::NCDHW,  InferenceEngine::Layout::NDHWC,   InferenceEngine::Layout::OIHW,
        InferenceEngine::Layout::GOIHW,  InferenceEngine::Layout::OIDHW,   InferenceEngine::Layout::GOIDHW,
        InferenceEngine::Layout::SCALAR, InferenceEngine::Layout::C,       InferenceEngine::Layout::CHW,
        InferenceEngine::Layout::HWC,    InferenceEngine::Layout::HW,      InferenceEngine::Layout::NC,
        InferenceEngine::Layout::CN,     InferenceEngine::Layout::BLOCKED,
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestIOBBlobSetLayoutTest,
                         ::testing::Combine(::testing::ValuesIn(layouts),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         InferRequestIOBBlobSetLayoutTestName::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestIOBBlobSetLayoutTest,
                         ::testing::Combine(::testing::ValuesIn(layouts),
                                            ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoconfigs)),
                         InferRequestIOBBlobSetLayoutTestName::getTestCaseName);

}  // namespace
