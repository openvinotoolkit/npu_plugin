// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <base/ov_behavior_test_utils.hpp>
#include <cstring>
#include <details/ie_exception.hpp>
#include <ie_core.hpp>
#include <string>
#include <vector>
#include <vpux/al/config/compiler.hpp>
#include "common/functions.h"
#include "common/vpu_test_env_cfg.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux_private_properties.hpp"

using CompilerType = ov::intel_vpux::CompilerType;

namespace {

class NPUTestCompiledModel :
        public ov::test::behavior::OVPluginTestBase,
        public testing::WithParamInterface<std::tuple<std::string, ov::AnyMap>> {
public:
    void SetUp() override {
        std::tie(target_device, configuration) = GetParam();
        OVPluginTestBase::SetUp();
    }

    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<std::string, ov::AnyMap>> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << LayerTestsUtils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            using namespace ov::test::utils;
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

protected:
    ov::AnyMap configuration;
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
};

TEST_P(NPUTestCompiledModel, samePlatformProduceTheSameBlob) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        std::string platform = LayerTestsUtils::getTestsPlatformFromEnvironmentOr("3700");

        configuration[ov::intel_vpux::create_executor.name()] = "0";
        auto configuration1 = configuration;
        configuration1[ov::intel_vpux::platform.name()] = platform;
        const auto& ov_model1 = buildSingleLayerSoftMaxNetwork();
        auto compiled_model1 = core->compile_model(ov_model1, target_device, configuration1);
        std::stringstream blobStream1;
        compiled_model1.export_model(blobStream1);

        auto configuration2 = configuration;
        configuration2[ov::intel_vpux::platform.name()] = platform;
        const auto& ov_model2 = buildSingleLayerSoftMaxNetwork();
        auto compiled_model2 = core->compile_model(ov_model2, target_device, configuration2);
        std::stringstream blobStream2;
        compiled_model2.export_model(blobStream2);

        ASSERT_NE(0, blobStream1.str().size());
        ASSERT_EQ(0, std::memcmp(blobStream1.str().c_str(), blobStream2.str().c_str(), blobStream1.str().size()));
    }
}

class NPUTestCompileModelWithoutDevice : public NPUTestCompiledModel {
protected:
    void SetUp() override {
        const auto devices = core->get_available_devices();
        const auto isNPUDeviceAvailable =
                std::find_if(devices.cbegin(), devices.cend(), [this](const std::string& device) {
                    return device.find(target_device) != std::string::npos;
                }) != devices.cend();
        if (isNPUDeviceAvailable) {
            GTEST_SKIP() << "Skip the tests since device is available";
        }
    }
};

TEST_P(NPUTestCompileModelWithoutDevice, ThrowIfNoDeviceAndNoPlatform) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        const auto& ov_model = buildSingleLayerSoftMaxNetwork();
        ASSERT_THROW(auto compiled_model = core->compile_model(ov_model, target_device, configuration), ov::Exception);
    }
}

TEST_P(NPUTestCompileModelWithoutDevice, NoThrowIfNoDeviceAndButPlatformPassed) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        auto netConfiguration = configuration;
        netConfiguration[ov::intel_vpux::platform.name()] = LayerTestsUtils::getTestsPlatformFromEnvironmentOr("3700");
        const auto& ov_model = buildSingleLayerSoftMaxNetwork();
        ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_model, target_device, netConfiguration));
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

TEST_P(NPUTestCompileModelWithoutDevice, CheckDeviceInBlob) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        // Compile model to target plugins, wrong platform specified -> expect an exception
        auto netConfigurationMLIR_wrong = configuration;
        netConfigurationMLIR_wrong[ov::intel_vpux::platform.name()] =
                getWrongDevice(PlatformEnvironment::PLATFORM, CompilerType::MLIR);
        netConfigurationMLIR_wrong[ov::intel_vpux::compiler_type.name()] = "MLIR";
        const auto& ov_model1 = buildSingleLayerSoftMaxNetwork();
        EXPECT_ANY_THROW(auto compiled_model =
                                 core->compile_model(ov_model1, target_device, netConfigurationMLIR_wrong));

        // Compile model to target plugins, valid platform specified -> expect no exception
        auto netConfigurationMLIR_valid = configuration;
        netConfigurationMLIR_valid[ov::intel_vpux::platform.name()] =
                getValidDevice(PlatformEnvironment::PLATFORM, CompilerType::MLIR);
        netConfigurationMLIR_valid[ov::intel_vpux::compiler_type.name()] = "MLIR";
        const auto& ov_model2 = buildSingleLayerSoftMaxNetwork();
        EXPECT_NO_THROW(auto compiled_model =
                                core->compile_model(ov_model2, target_device, netConfigurationMLIR_valid));
    }
}

}  // namespace
