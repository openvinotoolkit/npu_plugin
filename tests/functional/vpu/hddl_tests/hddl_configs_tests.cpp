// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <thread>
#include <chrono>
#include <map>

#include <gtest/gtest.h>

#include "ie_common.h"
#include "tests_vpu_common.hpp"
#include "myriad_layers_tests.hpp"

#include "plugin_cache.hpp"

#include "tests_hddl_utils.hpp"

using config_t = std::map<std::string, std::string>;

static const std::string modelName = {ModelsPath() + "/conv_conv_med/conv_conv_med_fp16"};

// TODO: askrebko: actually, the tests below are behavior tests but they depends on external configuration
// so let's keep them here until hddl-service allows to reconfigure itself in runtime.

class hddlCorrectConfigsTests_nightly: public myriadLayerTestBaseWithParam<config_t> {
protected:
    void SetUp() override {
#if defined(_WIN32) || defined(WIN32)
        return;
#endif
        myriadLayersTests_nightly::SetUp();
        HDDLTestsUtils::killHddlService();

        const auto &config = GetParam();
        std::string hddlServiceConfigFilename = getConfigFileCorrespondingToOption(config);
        ASSERT_TRUE(HDDLTestsUtils::runHddlService(hddlServiceConfigFilename));
    }

    void TearDown() override {
        myriadLayersTests_nightly::TearDown();

        ASSERT_TRUE(HDDLTestsUtils::killHddlService());
    }

    virtual std::string getConfigFileCorrespondingToOption(const std::map<std::string, std::string> &config) const {
        std::string hddlServiceConfigFilename;

        if (config.find("VPU_HDDL_GRAPH_TAG") != config.end()) {
            hddlServiceConfigFilename = "hddl_service_with_graph_map.config";
        } else if (config.find("VPU_HDDL_STREAM_ID") != config.end()) {
            hddlServiceConfigFilename = "hddl_service_with_stream_device_number_4.config";
        } else if (config.find("VPU_HDDL_DEVICE_TAG") != config.end()) {
            hddlServiceConfigFilename = "hddl_service_bypass_device_number_4.config";
        }

        return hddlServiceConfigFilename;
    }
};

class hddlIncorrectConfigsTests_nightly: public hddlCorrectConfigsTests_nightly {
protected:
    std::string getConfigFileCorrespondingToOption(const std::map<std::string, std::string> &config) const override {
        return "hddl_service_incorrect.config";
    }
};

TEST_P(hddlCorrectConfigsTests_nightly, SetCorrectConfig) {
    DISABLE_IF(!hasHDDL_R());
#if defined(_WIN32) || defined(WIN32)
        SKIP() << "Disabled since hddldaemon cannot be stopped on windows properly" << std::endl;
#endif

    InferenceEngine::ResponseDesc response;

    InferenceEngine::StatusCode sts = myriadPluginPtr->SetConfig(GetParam(), &response);

    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts) << response.msg;

    PluginCache::get().reset();
}

TEST_P(hddlIncorrectConfigsTests_nightly, SetIncorrectConfig) {
    DISABLE_IF(!hasHDDL_R());
#if defined(_WIN32) || defined(WIN32)
        SKIP() << "Disabled since hddldaemon cannot be stopped on windows properly" << std::endl;
#endif
    InferenceEngine::ResponseDesc response;

    InferenceEngine::StatusCode sts = myriadPluginPtr->SetConfig(GetParam(), &response);

    ASSERT_EQ(InferenceEngine::StatusCode::GENERAL_ERROR, sts) << response.msg;

    PluginCache::get().reset();
}

TEST_P(hddlCorrectConfigsTests_nightly, LoadNetworkWithCorrectConfig) {
    DISABLE_IF(!hasHDDL_R());
#if defined(_WIN32) || defined(WIN32)
        SKIP() << "Disabled since hddldaemon cannot be stopped on windows properly" << std::endl;
#endif
    InferenceEngine::ResponseDesc response;
    const auto &config = GetParam();

    InferenceEngine::CNNNetReader reader;
    ASSERT_NO_THROW(reader.ReadNetwork(modelName + ".xml"));
    ASSERT_NO_THROW(reader.ReadWeights(modelName + ".bin"));
    InferenceEngine::IExecutableNetwork::Ptr executable;
    InferenceEngine::StatusCode sts = myriadPluginPtr->LoadNetwork(executable, reader.getNetwork(), config, &response);

    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts) << response.msg;
}

TEST_P(hddlIncorrectConfigsTests_nightly, LoadNetworkWithIncorrectConfig) {
    DISABLE_IF(!hasHDDL_R());
#if defined(_WIN32) || defined(WIN32)
        SKIP() << "Disabled since hddldaemon cannot be stopped on windows properly" << std::endl;
#endif
    InferenceEngine::ResponseDesc response;
    const auto &config = GetParam();

    InferenceEngine::CNNNetReader reader;
    ASSERT_NO_THROW(reader.ReadNetwork(modelName + ".xml"));
    ASSERT_NO_THROW(reader.ReadWeights(modelName + ".bin"));
    InferenceEngine::IExecutableNetwork::Ptr executable;
    InferenceEngine::StatusCode sts = myriadPluginPtr->LoadNetwork(executable, reader.getNetwork(), config, &response);

    ASSERT_EQ(InferenceEngine::StatusCode::GENERAL_ERROR, sts) << response.msg;
}

static const std::vector<config_t> hddlConfigCorrect = {
    { {VPU_HDDL_CONFIG_KEY(GRAPH_TAG), "graphTag"} },
    { {VPU_HDDL_CONFIG_KEY(STREAM_ID), "sid"} },
    { {VPU_HDDL_CONFIG_KEY(DEVICE_TAG), "deviceTag"} },
    { {VPU_HDDL_CONFIG_KEY(DEVICE_TAG), "deviceTag"}, {VPU_HDDL_CONFIG_KEY(BIND_DEVICE), CONFIG_VALUE(YES)} },
    { {VPU_HDDL_CONFIG_KEY(DEVICE_TAG), "deviceTag"}, {VPU_HDDL_CONFIG_KEY(BIND_DEVICE), CONFIG_VALUE(NO)}},
    { {VPU_HDDL_CONFIG_KEY(DEVICE_TAG), "deviceTag"}, {VPU_HDDL_CONFIG_KEY(BIND_DEVICE), CONFIG_VALUE(YES)},
      {VPU_HDDL_CONFIG_KEY(RUNTIME_PRIORITY), "10"} },
    { {VPU_HDDL_CONFIG_KEY(DEVICE_TAG), "deviceTag"}, {VPU_HDDL_CONFIG_KEY(BIND_DEVICE), CONFIG_VALUE(NO)},
      {VPU_HDDL_CONFIG_KEY(RUNTIME_PRIORITY), "10"} },
    { {VPU_HDDL_CONFIG_KEY(DEVICE_TAG), "deviceTag"}, {VPU_HDDL_CONFIG_KEY(BIND_DEVICE), CONFIG_VALUE(NO)} }
};

static const std::vector<config_t> hddlConfigIncorrect = {
    { {VPU_HDDL_CONFIG_KEY(GRAPH_TAG), "graphTag"}, {VPU_HDDL_CONFIG_KEY(STREAM_ID), "sid"} },
    { {VPU_HDDL_CONFIG_KEY(GRAPH_TAG), "graphTag"}, {VPU_HDDL_CONFIG_KEY(DEVICE_TAG), "deviceTag"} },
    { {VPU_HDDL_CONFIG_KEY(STREAM_ID), "sid"}, {VPU_HDDL_CONFIG_KEY(DEVICE_TAG), "deviceTag"} },
    { {VPU_HDDL_CONFIG_KEY(BIND_DEVICE), CONFIG_VALUE(YES)} },
    { {VPU_HDDL_CONFIG_KEY(BIND_DEVICE), CONFIG_VALUE(NO)} },
    { {VPU_HDDL_CONFIG_KEY(RUNTIME_PRIORITY), "10"} },
    { {VPU_HDDL_CONFIG_KEY(BIND_DEVICE), CONFIG_VALUE(YES)},{VPU_HDDL_CONFIG_KEY(RUNTIME_PRIORITY), "10"} },
    { {VPU_HDDL_CONFIG_KEY(BIND_DEVICE), CONFIG_VALUE(NO)}, {VPU_HDDL_CONFIG_KEY(RUNTIME_PRIORITY), "10"} },
    { {VPU_HDDL_CONFIG_KEY(DEVICE_TAG), "deviceTag"}, {VPU_HDDL_CONFIG_KEY(BIND_DEVICE), CONFIG_VALUE(NO)},
      {VPU_HDDL_CONFIG_KEY(RUNTIME_PRIORITY), "abc"} },
    { {VPU_HDDL_CONFIG_KEY(DEVICE_TAG), "deviceTag"}, {VPU_HDDL_CONFIG_KEY(BIND_DEVICE), "ON"},
      {VPU_HDDL_CONFIG_KEY(RUNTIME_PRIORITY), "10"} }
};

INSTANTIATE_TEST_CASE_P(HDDLConfigs, hddlCorrectConfigsTests_nightly,   ::testing::ValuesIn(hddlConfigCorrect));
INSTANTIATE_TEST_CASE_P(HDDLConfigs, hddlIncorrectConfigsTests_nightly, ::testing::ValuesIn(hddlConfigIncorrect));

using hddlCorrectConfigsUpdateRuntimePriorityTests_nightly = hddlCorrectConfigsTests_nightly;

TEST_P(hddlCorrectConfigsUpdateRuntimePriorityTests_nightly, CanChangeRuntimePriority) {
    DISABLE_IF(!hasHDDL_R());
#if defined(_WIN32) || defined(WIN32)
        SKIP() << "Disabled since hddldaemon cannot be stopped on windows properly" << std::endl;
#endif
    InferenceEngine::ResponseDesc response;
    const std::map<std::string, std::string> defaultConfig = { {VPU_HDDL_CONFIG_KEY(DEVICE_TAG), "deviceTag"},
                                                               {VPU_HDDL_CONFIG_KEY(BIND_DEVICE), CONFIG_VALUE(NO)} };
    const auto &configWithNewPriority = GetParam();

    InferenceEngine::CNNNetReader reader;
    ASSERT_NO_THROW(reader.ReadNetwork(modelName + ".xml"));
    ASSERT_NO_THROW(reader.ReadWeights(modelName + ".bin"));
    InferenceEngine::IExecutableNetwork::Ptr executable;
    ASSERT_EQ(InferenceEngine::StatusCode::OK, myriadPluginPtr->LoadNetwork(executable, reader.getNetwork(), defaultConfig, &response)) << response.msg;

    ASSERT_EQ(InferenceEngine::StatusCode::OK, myriadPluginPtr->LoadNetwork(executable, reader.getNetwork(), configWithNewPriority, &response)) << response.msg;
}

using hddlIncorrectUsageConfigsUpdateRuntimePriorityTests_nightly = hddlCorrectConfigsTests_nightly;

TEST_P(hddlIncorrectUsageConfigsUpdateRuntimePriorityTests_nightly, TryingToSet) {
    DISABLE_IF(!hasHDDL_R());
#if defined(_WIN32) || defined(WIN32)
        SKIP() << "Disabled since hddldaemon cannot be stopped on windows properly" << std::endl;
#endif
    InferenceEngine::ResponseDesc response;
    const std::map<std::string, std::string> defaultConfig = { {VPU_HDDL_CONFIG_KEY(DEVICE_TAG), "deviceTag"},
                                                               {VPU_HDDL_CONFIG_KEY(BIND_DEVICE), CONFIG_VALUE(NO)} };
    const auto &configWithNewPriority = GetParam();

    InferenceEngine::CNNNetReader reader;
    ASSERT_NO_THROW(reader.ReadNetwork(modelName + ".xml"));
    ASSERT_NO_THROW(reader.ReadWeights(modelName + ".bin"));
    InferenceEngine::IExecutableNetwork::Ptr executable;
    ASSERT_EQ(InferenceEngine::StatusCode::OK, myriadPluginPtr->LoadNetwork(executable, reader.getNetwork(), defaultConfig, &response)) << response.msg;

    ASSERT_EQ(InferenceEngine::StatusCode::GENERAL_ERROR, myriadPluginPtr->LoadNetwork(executable, reader.getNetwork(), configWithNewPriority, &response)) << response.msg;
}
