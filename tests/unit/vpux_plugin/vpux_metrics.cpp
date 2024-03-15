//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include <gmock/gmock-matchers.h>

#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"
#include "vpux/vpux_plugin_params.hpp"
#include "vpux_metrics.hpp"

namespace ie = InferenceEngine;

using MetricsUnitTests = ::testing::Test;
using ::testing::HasSubstr;

TEST_F(MetricsUnitTests, getAvailableDevicesNames) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3700_test_backend"};
    vpux::VPUXBackends::Ptr backends;
    backends = std::make_shared<vpux::VPUXBackends>(dummyBackendRegistry, dummyConfig);
    vpux::Metrics metrics(backends);

    std::vector<std::string> devicesNames = metrics.GetAvailableDevicesNames();

    ASSERT_EQ("DummyVPU3700Device", devicesNames[0]);
    ASSERT_EQ("noOtherDevice", devicesNames[1]);
}

TEST_F(MetricsUnitTests, getFullDeviceName) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3720_test_backend"};
    vpux::VPUXBackends::Ptr backends;
    backends = std::make_shared<vpux::VPUXBackends>(dummyBackendRegistry, dummyConfig);
    vpux::Metrics metrics(backends);

    auto device = backends->getDevice();
    EXPECT_THAT(metrics.GetFullDeviceName(device->getName()), HasSubstr("Intel(R) NPU"));
}

TEST_F(MetricsUnitTests, getDeviceUuid) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3720_test_backend"};
    vpux::VPUXBackends::Ptr backends;
    backends = std::make_shared<vpux::VPUXBackends>(dummyBackendRegistry, dummyConfig);
    vpux::Metrics metrics(backends);

    ov::device::UUID testPattern = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x37, 0x20};

    auto device = backends->getDevice();
    ov::device::UUID getDeviceUuid = metrics.GetDeviceUuid(device->getName());

    for (uint64_t i = 0; i < getDeviceUuid.MAX_UUID_SIZE; i++) {
        ASSERT_EQ(testPattern.uuid[i], getDeviceUuid.uuid[i]);
    }
}

TEST_F(MetricsUnitTests, getDeviceArchitecture) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3720_test_backend"};
    vpux::VPUXBackends::Ptr backends;
    backends = std::make_shared<vpux::VPUXBackends>(dummyBackendRegistry, dummyConfig);
    vpux::Metrics metrics(backends);

    auto device = backends->getDevice();
    ASSERT_EQ("3720", metrics.GetDeviceArchitecture(device->getName()));
}

TEST_F(MetricsUnitTests, getBackendName) {
    std::shared_ptr<vpux::OptionsDesc> dummyOptions = std::make_shared<vpux::OptionsDesc>();
    vpux::Config dummyConfig(dummyOptions);
    const std::vector<std::string> dummyBackendRegistry = {"vpu3720_test_backend"};
    vpux::VPUXBackends::Ptr backends;
    backends = std::make_shared<vpux::VPUXBackends>(dummyBackendRegistry, dummyConfig);
    vpux::Metrics metrics(backends);

    ASSERT_EQ("VPU3720TestBackend", metrics.GetBackendName());
}
