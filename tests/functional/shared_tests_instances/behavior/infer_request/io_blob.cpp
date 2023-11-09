//
// Copyright (C) 2018-2021 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <vector>

#include "behavior/infer_request/io_blob.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<std::map<std::string, std::string>> configs = {{}};

const std::vector<std::map<std::string, std::string>> autoconfigs = {
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_KEEMBAY}},
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
          std::string(CommonTestUtils::DEVICE_CPU) + "," + CommonTestUtils::DEVICE_KEEMBAY}}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestIOBBlobTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         InferRequestIOBBlobTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestIOBBlobTest,
                         ::testing::Combine(::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoconfigs)),
                         InferRequestIOBBlobTest::getTestCaseName);

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
                                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         InferRequestIOBBlobSetLayoutTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, InferRequestIOBBlobSetLayoutTest,
                         ::testing::Combine(::testing::ValuesIn(layouts),
                                            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoconfigs)),
                         InferRequestIOBBlobSetLayoutTest::getTestCaseName);

}  // namespace
