//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <base/behavior_test_utils.hpp>
#include <details/ie_exception.hpp>
#include <ie_core.hpp>
#include <string>
#include <vector>
#include <vpux/al/config/compiler.hpp>
#include "common/functions.h"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "vpu_test_env_cfg.hpp"
#include "vpux_private_config.hpp"

using LoadNetwork = BehaviorTestsUtils::BehaviorTestsBasic;
using CompilerType = InferenceEngine::VPUXConfigParams::CompilerType;

namespace {

TEST_P(LoadNetwork, samePlatformProduceTheSameBlob) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        std::string platform = LayerTestsUtils::getTestsPlatformFromEnvironmentOr("3700");
        if (platform.find("_EMU") != std::string::npos)
            GTEST_SKIP() << "Export() not supported for emulator platform.";

        auto cnnNet = buildSingleLayerSoftMaxNetwork();

        configuration["NPU_CREATE_EXECUTOR"] = "0";
        auto configuration1 = configuration;
        configuration1[VPUX_CONFIG_KEY(PLATFORM)] = platform;
        auto exeNet1 = ie->LoadNetwork(cnnNet, "NPU", configuration1);
        std::stringstream blobStream1;
        exeNet1.Export(blobStream1);

        auto configuration2 = configuration;
        configuration2[VPUX_CONFIG_KEY(PLATFORM)] = platform;
        auto exeNet2 = ie->LoadNetwork(cnnNet, "NPU", configuration2);
        std::stringstream blobStream2;
        exeNet2.Export(blobStream2);

        ASSERT_NE(0, blobStream1.str().size());
        ASSERT_EQ(blobStream1.str(), blobStream2.str());
    }
}

class LoadNetworkWithoutDevice : public LoadNetwork {
protected:
    void SetUp() override {
        const auto devices = ie->GetAvailableDevices();
        const auto isVPUXDeviceAvailable =
                std::find_if(devices.cbegin(), devices.cend(), [](const std::string& device) {
                    return device.find("NPU") != std::string::npos;
                }) != devices.cend();
        if (isVPUXDeviceAvailable) {
            GTEST_SKIP() << "Skip the tests since device is available";
        }
    }
};

TEST_P(LoadNetworkWithoutDevice, ThrowIfNoDeviceAndNoPlatform) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto cnnNet = buildSingleLayerSoftMaxNetwork();
        ASSERT_THROW(ie->LoadNetwork(cnnNet, "NPU", configuration), InferenceEngine::Exception);
    }
}

TEST_P(LoadNetworkWithoutDevice, NoThrowIfNoDeviceAndButPlatformPassed) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto cnnNet = buildSingleLayerSoftMaxNetwork();
        auto netConfiguration = configuration;
        netConfiguration[VPUX_CONFIG_KEY(PLATFORM)] = LayerTestsUtils::getTestsPlatformFromEnvironmentOr("3700");
        ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, "NPU", netConfiguration));
    }
}

const std::map<std::string, std::array<std::string, 2>> wrongDevice = {
        // {orig, {wrong for MLIR}}
        {"VPU3700", {"VPU0000"}},
        {"VPU3720", {"VPU0000"}},
};

std::string getWrongDevice(const std::string& platform, const CompilerType&) {
    // here we mix up devices in order to test the check on the runtime side
    auto device = wrongDevice.find(platform);

    if (device == wrongDevice.end())
        THROW_IE_EXCEPTION << "Cannot map wrong device for the platform " << platform;
    return device->second[0];
}

const std::map<std::string, std::array<std::string, 2>> validDevice = {
        // {orig, {valid for MLIR}}
        {"VPU3700", {"VPU3700"}},
        {"VPU3720", {"VPU3720"}},
};

std::string getValidDevice(const std::string& platform, const CompilerType&) {
    auto device = validDevice.find(platform);

    if (device == validDevice.end())
        THROW_IE_EXCEPTION << "Cannot map valid device for the platform " << platform;
    return device->second[0];
}

TEST_P(LoadNetwork, CheckDeviceInBlob) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        InferenceEngine::CNNNetwork cnnNet = buildSingleLayerSoftMaxNetwork();
        // Load CNNNetwork to target plugins, wrong platform specified -> expect an exception
        auto netConfigurationMLIR_wrong = configuration;
        netConfigurationMLIR_wrong[VPUX_CONFIG_KEY(PLATFORM)] =
                getWrongDevice(PlatformEnvironment::PLATFORM, CompilerType::MLIR);
        netConfigurationMLIR_wrong[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);
        EXPECT_ANY_THROW(ie->LoadNetwork(cnnNet, target_device, netConfigurationMLIR_wrong));

        // Load CNNNetwork to target plugins, valid platform specified -> expect no exception
        auto netConfigurationMLIR_valid = configuration;
        netConfigurationMLIR_valid[VPUX_CONFIG_KEY(PLATFORM)] =
                getValidDevice(PlatformEnvironment::PLATFORM, CompilerType::MLIR);
        netConfigurationMLIR_valid[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);
        EXPECT_NO_THROW(ie->LoadNetwork(cnnNet, target_device, netConfigurationMLIR_valid));
    }
}

}  // namespace
