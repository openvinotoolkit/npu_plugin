//
// Copyright 2020-2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <gtest/gtest.h>

#include <vpux_backends.h>

namespace ie = InferenceEngine;

using VPUXBackendsUnitTests = ::testing::Test;

// [Track number: E#9567]
TEST_F(VPUXBackendsUnitTests, DISABLED_notStopSearchingIfBackendThrow) {
    const std::vector<std::string> dummyBackendRegistry = {"throw_test_backend", "one_device_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry);

    vpux::VPUXConfig config;
    config.update({{"LOG_LEVEL", "LOG_DEBUG"}});
    backends.setup(config);

    auto device = backends.getDevice();
    ASSERT_NE(nullptr, device);
    ASSERT_EQ("DummyDevice", device->getName());
}

// [Track number: E#9567]
TEST_F(VPUXBackendsUnitTests, DISABLED_canFindDeviceIfAtLeastOneBackendHasDevicesAvailable) {
    const std::vector<std::string> dummyBackendRegistry = {"no_devices_test_backend", "one_device_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry);

    auto device = backends.getDevice();
    ASSERT_NE(nullptr, device);
    ASSERT_EQ("DummyDevice", device->getName());
}

TEST_F(VPUXBackendsUnitTests, deviceReturnsNullptrIfNoBackends) {
    vpux::VPUXBackends backends({});
    ASSERT_EQ(nullptr, backends.getDevice());
}

TEST_F(VPUXBackendsUnitTests, deviceReturnsNullptrIfPassedBackendsNotExist) {
    vpux::VPUXBackends backends({"wrong_path", "one_more_wrong_path"});
    ASSERT_EQ(nullptr, backends.getDevice());
}
