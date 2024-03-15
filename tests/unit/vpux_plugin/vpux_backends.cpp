//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"
#include "vpux/vpux_plugin_params.hpp"
#include "vpux_backends.hpp"

namespace ie = InferenceEngine;

using VPUXBackendsUnitTests = ::testing::Test;

TEST_F(VPUXBackendsUnitTests, notStopSearchingIfBackendThrow) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"throw_test_backend", "vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry, dummyConfig);

    auto options = std::make_shared<vpux::OptionsDesc>();
    vpux::registerCommonOptions(*options);
    vpux::registerRunTimeOptions(*options);

    vpux::Config config(options);
    config.update({{"LOG_LEVEL", "LOG_DEBUG"}});

    backends.setup(config);

    auto device = backends.getDevice();
    ASSERT_NE(nullptr, device);
    ASSERT_EQ("DummyVPU3700Device", device->getName());
}

TEST_F(VPUXBackendsUnitTests, notStopSearchingIfBackendNotExists) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"not_exists_backend", "vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry, dummyConfig);

    auto options = std::make_shared<vpux::OptionsDesc>();
    vpux::registerCommonOptions(*options);
    vpux::registerRunTimeOptions(*options);

    vpux::Config config(options);
    config.update({{"LOG_LEVEL", "LOG_DEBUG"}});

    backends.setup(config);

    auto device = backends.getDevice();
    ASSERT_NE(nullptr, device);
    ASSERT_EQ("DummyVPU3700Device", device->getName());
}

TEST_F(VPUXBackendsUnitTests, canFindDeviceIfAtLeastOneBackendHasDevicesAvailable) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"no_devices_test_backend", "vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry, dummyConfig);

    auto device = backends.getDevice();
    ASSERT_NE(nullptr, device);
    ASSERT_EQ("DummyVPU3700Device", device->getName());
}

TEST_F(VPUXBackendsUnitTests, deviceReturnsNullptrIfNoBackends) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    vpux::VPUXBackends backends({}, dummyConfig);
    ASSERT_EQ(nullptr, backends.getDevice());
}

TEST_F(VPUXBackendsUnitTests, deviceReturnsNullptrIfPassedBackendsNotExist) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    vpux::VPUXBackends backends({"wrong_path", "one_more_wrong_path"}, dummyConfig);
    ASSERT_EQ(nullptr, backends.getDevice());
}

TEST_F(VPUXBackendsUnitTests, findDeviceAfterName) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry, dummyConfig);

    std::string deviceName = "DummyVPU3700Device";
    auto device = backends.getDevice(deviceName);
    ASSERT_NE(nullptr, device);
    ASSERT_EQ(deviceName, device->getName());
}

TEST_F(VPUXBackendsUnitTests, noDeviceFoundedAfterName) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry, dummyConfig);

    std::string deviceName = "wrong_device";
    ASSERT_EQ(nullptr, backends.getDevice(deviceName));
}

TEST_F(VPUXBackendsUnitTests, findDeviceAfterParamMap) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry, dummyConfig);

    ie::ParamMap paramMap = {{ie::VPUX_PARAM_KEY(DEVICE_ID), 3700}};
    auto device = backends.getDevice(paramMap);
    ASSERT_NE(nullptr, device);
    ASSERT_EQ("DummyVPU3700Device", device->getName());
}

TEST_F(VPUXBackendsUnitTests, noDeviceFoundedAfterParamMap) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry, dummyConfig);

    ie::ParamMap paramMap = {{ie::VPUX_PARAM_KEY(DEVICE_ID), 3000}};
    ASSERT_EQ(nullptr, backends.getDevice(paramMap));
}

TEST_F(VPUXBackendsUnitTests, getDeviceNamesSecondIsDummyName) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry, dummyConfig);

    std::vector<std::string> deviceNames = backends.getAvailableDevicesNames();

    ASSERT_EQ("DummyVPU3700Device", deviceNames[0]);
    ASSERT_EQ("noOtherDevice", deviceNames[1]);
}

TEST_F(VPUXBackendsUnitTests, getBackendName) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry, dummyConfig);

    std::string backendName = backends.getBackendName();

    ASSERT_EQ("VPU3700TestBackend", backendName);
}

TEST_F(VPUXBackendsUnitTests, getCompilationPlatformByDeviceName) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3720_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry, dummyConfig);

    std::string compilationPlatform =
            backends.getCompilationPlatform(ie::VPUXConfigParams::VPUXPlatform::AUTO_DETECT, "");

    ASSERT_EQ("3720", compilationPlatform);
}

TEST_F(VPUXBackendsUnitTests, getCompilationPlatformByPlatform) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry, dummyConfig);

    std::string compilationPlatform3700 =
            backends.getCompilationPlatform(ie::VPUXConfigParams::VPUXPlatform::VPU3700, "");
    std::string compilationPlatform3720 =
            backends.getCompilationPlatform(ie::VPUXConfigParams::VPUXPlatform::VPU3720, "");

    ASSERT_EQ("3700", compilationPlatform3700);
    ASSERT_EQ("3720", compilationPlatform3720);
}

TEST_F(VPUXBackendsUnitTests, getCompilationPlatformByDeviceId) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry, dummyConfig);

    std::string compilationPlatform3700 =
            backends.getCompilationPlatform(ie::VPUXConfigParams::VPUXPlatform::AUTO_DETECT, "3700");
    std::string compilationPlatform3720 =
            backends.getCompilationPlatform(ie::VPUXConfigParams::VPUXPlatform::AUTO_DETECT, "3720");

    ASSERT_EQ("3700", compilationPlatform3700);
    ASSERT_EQ("3720", compilationPlatform3720);
}

TEST_F(VPUXBackendsUnitTests, getCompilationPlatformByDeviceNameNoDevice) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"no_devices_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry, dummyConfig);

    try {
        std::string compilationPlatform =
                backends.getCompilationPlatform(ie::VPUXConfigParams::VPUXPlatform::AUTO_DETECT, "");
    } catch (const std::exception& ex) {
        std::string expectedMessage("No devices found - platform must be explicitly specified for compilation. "
                                    "Example: -d NPU.3700 instead of -d NPU.\n");
        std::string exceptionMessage(ex.what());
        // exception message contains information about path to file and line number, where the exception occurred.
        // We should ignore this part of the message on comparision step
        ASSERT_TRUE(exceptionMessage.length() >= expectedMessage.length());
        ASSERT_EQ(expectedMessage, exceptionMessage.substr(exceptionMessage.length() - expectedMessage.length()));
    } catch (...) {
        FAIL() << "UNEXPECTED RESULT";
    }
}

TEST_F(VPUXBackendsUnitTests, getCompilationPlatformByDeviceNameWrongNameFormat) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry, dummyConfig);

    try {
        std::string compilationPlatform =
                backends.getCompilationPlatform(ie::VPUXConfigParams::VPUXPlatform::AUTO_DETECT, "");
    } catch (const std::exception& ex) {
        std::string expectedMessage("Unexpected device name: DummyVPU3700Device\n");
        std::string exceptionMessage(ex.what());
        // exception message contains information about path to file and line number, where the exception occurred.
        // We should ignore this part of the message on comparision step
        ASSERT_TRUE(exceptionMessage.length() >= expectedMessage.length());
        ASSERT_EQ(expectedMessage, exceptionMessage.substr(exceptionMessage.length() - expectedMessage.length()));
    } catch (...) {
        FAIL() << "UNEXPECTED RESULT";
    }
}
