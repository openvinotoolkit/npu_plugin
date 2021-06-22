// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <ie_core.hpp>
#include <base/behavior_test_utils.hpp>
#include "vpux_private_config.hpp"
#include "common/functions.h"
#include <details/ie_exception.hpp>


using LoadNetwork = BehaviorTestsUtils::BehaviorTestsBasic;

namespace {


TEST_P(LoadNetwork, samePlatformProduceTheSameBlob) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    {
        auto cnnNet = buildSingleLayerSoftMaxNetwork();

        auto configuration1 = configuration;
        configuration1[VPUX_CONFIG_KEY(PLATFORM)] = PlatformEnvironment::PLATFORM;
        auto exeNet1 = ie->LoadNetwork(cnnNet, "VPUX", configuration1);
        std::stringstream blobStream1;
        exeNet1.Export(blobStream1);

        auto configuration2 = configuration;
        configuration2[VPUX_CONFIG_KEY(PLATFORM)] = PlatformEnvironment::PLATFORM;
        auto exeNet2 = ie->LoadNetwork(cnnNet, "VPUX", configuration2);
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
        const auto isVPUXDeviceAvailable = std::find_if(devices.cbegin(), devices.cend(), [](const std::string& device) {
                return device.find("VPUX") != std::string::npos;
            }) != devices.cend();
        if (isVPUXDeviceAvailable) {
            GTEST_SKIP() << "Skip the tests since device is available";
        }
    }
};

TEST_P(LoadNetworkWithoutDevice, ThrowIfNoDeviceAndNoPlatform) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    {
        auto cnnNet = buildSingleLayerSoftMaxNetwork();
        ASSERT_THROW(ie->LoadNetwork(cnnNet, "VPUX", configuration), InferenceEngine::Exception);
    }
}

TEST_P(LoadNetworkWithoutDevice, NoThrowIfNoDeviceAndButPlatformPassed) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    {
        auto cnnNet = buildSingleLayerSoftMaxNetwork();
        auto netConfiguration = configuration;
        netConfiguration[VPUX_CONFIG_KEY(PLATFORM)] = PlatformEnvironment::PLATFORM;
        ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, "VPUX", netConfiguration));
    }
}

const std::map<std::string, std::string> wrongDevice =
{
    {"3400_A0", "3400"},
    {"3400", "3400_A0"},
    {"3700", "3400_A0"},
    {"3720", "3400"},
    {"3900", "3400"},
    // For AUTO we can set 3400_A0 which is deprecated
    {"AUTO", "3400_A0"}
};

std::string getWrongDevice(const std::string& platform)
{
    // here we mix up devices in order to test the check on the runtime side
    auto device = wrongDevice.find(platform);

    if (device == wrongDevice.end())
        THROW_IE_EXCEPTION << "Cannot map wrong device for the platform " << platform;
    return device->second;
}

TEST_P(LoadNetwork, CheckDeviceInBlob) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    {
        InferenceEngine::CNNNetwork cnnNet = buildSingleLayerSoftMaxNetwork();
        // Load CNNNetwork to target plugins
        auto netConfiguration = configuration;
        netConfiguration[VPUX_CONFIG_KEY(PLATFORM)] = getWrongDevice(PlatformEnvironment::PLATFORM);
#if defined(__arm__) || defined(__aarch64__)
        EXPECT_ANY_THROW(ie->LoadNetwork(cnnNet, targetDevice, netConfiguration));
#else
        EXPECT_NO_THROW(ie->LoadNetwork(cnnNet, targetDevice, netConfiguration));
#endif
    }
}

}
