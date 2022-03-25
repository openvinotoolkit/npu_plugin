//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <gtest/gtest.h>

#include "vpux_backends.h"
#include "vpux/al/config/common.hpp"
#include "vpux/al/config/runtime.hpp"

namespace ie = InferenceEngine;

using VPUXBackendsUnitTests = ::testing::Test;

// [Track number: E#9567]
TEST_F(VPUXBackendsUnitTests, DISABLED_notStopSearchingIfBackendThrow) {
    const std::vector<std::string> dummyBackendRegistry = {"throw_test_backend", "one_device_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry);

    auto options = std::make_shared<vpux::OptionsDesc>();
    vpux::registerCommonOptions(*options);
    vpux::registerRunTimeOptions(*options);

    vpux::Config config(options);
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
