// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include <ie_core.hpp>
#include <base/behavior_test_utils.hpp>
#include "vpux/vpux_plugin_config.hpp"

#include <details/ie_exception.hpp>

using LoadNetwork = BehaviorTestsUtils::BehaviorTestsBasic;

namespace {

InferenceEngine::CNNNetwork createDummyNetwork() {
    InferenceEngine::SizeVector inputShape = {1, 3, 4, 3};
    InferenceEngine::Precision netPrecision = InferenceEngine::Precision::FP32;
    size_t axis = 1;

    const auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    const auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    const auto softMax = std::make_shared<ngraph::opset1::Softmax>(paramOuts.at(0), axis);

    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(softMax)};

    auto function = std::make_shared<ngraph::Function>(results, params, "softMax");
    InferenceEngine::CNNNetwork cnnNet(function);

    return cnnNet;
}

TEST_P(LoadNetwork, samePlatformProduceTheSameBlob) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    {
        auto cnnNet = createDummyNetwork();

        auto configuration1 = configuration;
        configuration1[VPUX_CONFIG_KEY(PLATFORM)] = VPUX_CONFIG_VALUE(VPU3400);
        auto exeNet1 = ie->LoadNetwork(cnnNet, "VPUX", configuration1);
        std::stringstream blobStream1;
        exeNet1.Export(blobStream1);

        auto configuration2 = configuration;
        configuration2[VPUX_CONFIG_KEY(PLATFORM)] = VPUX_CONFIG_VALUE(VPU3400);
        auto exeNet2 = ie->LoadNetwork(cnnNet, "VPUX", configuration2);
        std::stringstream blobStream2;
        exeNet2.Export(blobStream2);

        ASSERT_NE(0, blobStream1.str().size());
        ASSERT_EQ(blobStream1.str(), blobStream2.str());
    }
}

TEST_P(LoadNetwork, differentPlatformsProduceTheDifferentBlob) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    {
        auto cnnNet = createDummyNetwork();

        auto configuration1 = configuration;
        configuration1[VPUX_CONFIG_KEY(PLATFORM)] = VPUX_CONFIG_VALUE(VPU3400);
        auto exeNet1 = ie->LoadNetwork(cnnNet, "VPUX", configuration1);
        std::stringstream blobStream1;
        exeNet1.Export(blobStream1);

        auto configuration2 = configuration;
        configuration2[VPUX_CONFIG_KEY(PLATFORM)] = VPUX_CONFIG_VALUE(VPU3900);
        auto exeNet2 = ie->LoadNetwork(cnnNet, "VPUX", configuration2);
        std::stringstream blobStream2;
        exeNet2.Export(blobStream2);

        ASSERT_NE(0, blobStream1.str().size());
        ASSERT_NE(blobStream1.str(), blobStream2.str());
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
            SKIP() << "Skip the tests since device is available";
        }
    }
};

TEST_P(LoadNetworkWithoutDevice, ThrowIfNoDeviceAndNoPlatform) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    {
        auto cnnNet = createDummyNetwork();
        ASSERT_THROW(ie->LoadNetwork(cnnNet, "VPUX", configuration), InferenceEngine::Exception);
    }
}

TEST_P(LoadNetworkWithoutDevice, NoThrowIfNoDeviceAndButPlatformPassed) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    {
        auto cnnNet = createDummyNetwork();
        auto netConfiguration = configuration;
        netConfiguration[VPUX_CONFIG_KEY(PLATFORM)] = VPUX_CONFIG_VALUE(VPU3400);
        ASSERT_NO_THROW(ie->LoadNetwork(cnnNet, "VPUX", netConfiguration));
    }
}

}
