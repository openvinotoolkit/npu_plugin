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

TEST_F(VPUXBackendsUnitTests, notStopSearchingIfBackendThrow) {
    const std::vector<std::string> dummyBackendRegistry = {"throw_test_backend", "vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry);

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
    const std::vector<std::string> dummyBackendRegistry = {"not_exists_backend", "vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry);

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
    const std::vector<std::string> dummyBackendRegistry = {"no_devices_test_backend", "vpu3700_test_backend"};
    vpux::VPUXBackends backends(dummyBackendRegistry);

    auto device = backends.getDevice();
    ASSERT_NE(nullptr, device);
    ASSERT_EQ("DummyVPU3700Device", device->getName());
}

TEST_F(VPUXBackendsUnitTests, deviceReturnsNullptrIfNoBackends) {
    vpux::VPUXBackends backends({});
    ASSERT_EQ(nullptr, backends.getDevice());
}

TEST_F(VPUXBackendsUnitTests, deviceReturnsNullptrIfPassedBackendsNotExist) {
    vpux::VPUXBackends backends({"wrong_path", "one_more_wrong_path"});
    ASSERT_EQ(nullptr, backends.getDevice());
}
