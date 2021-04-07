//
// Copyright 2020-2021 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials, and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
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
