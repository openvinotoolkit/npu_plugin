// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <filesystem>
#include "behavior/ov_plugin/core_integration.hpp"
#include "common/utils.hpp"

namespace ov {
namespace test {
namespace behavior {

class OVClassBaseTestPVpux : public OVClassBaseTestP {
public:
    void SetUp() override {
        target_device = GetParam();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        APIBaseTest::SetUp();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        // Generic network
        actualNetwork = ngraph::builder::subgraph::makeSplitConvConcat();
        // Quite simple network
        simpleNetwork = ngraph::builder::subgraph::makeSingleConcatWithConstant();
        // Multinput to substruct network
        multinputNetwork = ngraph::builder::subgraph::makeConcatWithParams();
        // Network with KSO
        ksoNetwork = ngraph::builder::subgraph::makeKSOFunction();
    }
};

class OVClassBasicTestPVpux : public OVClassBasicTestP {
public:
    void TearDown() override {
        for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
            std::wstring postfix = ov::test::utils::test_unicode_postfix_vector[testIndex];
            std::wstring unicode_path = ov::test::utils::stringToWString(ov::util::get_ov_lib_path() + "/") + postfix;
#ifndef _WIN32
            removeDirFilesRecursive(ov::util::wstring_to_string(unicode_path));
#else
            removeDirFilesRecursive(unicode_path);
#endif
        }
    }
};

using OVClassNetworkTestPVpux = OVClassBaseTestPVpux;
using OVClassLoadNetworkTestVpux = OVClassBaseTestPVpux;

TEST_P(OVClassNetworkTestPVpux, LoadNetworkActualNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, target_device));
}

TEST_P(OVClassNetworkTestPVpux, LoadNetworkActualHeteroDeviceNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(
            ie.compile_model(actualNetwork, ov::test::utils::DEVICE_HETERO + std::string(":") + target_device));
}

TEST_P(OVClassNetworkTestPVpux, LoadNetworkActualHeteroDevice2NoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(
            ie.compile_model(actualNetwork, ov::test::utils::DEVICE_HETERO, ov::device::priorities(target_device)));
}

TEST_P(OVClassNetworkTestPVpux, LoadNetworkActualHeteroDeviceUsingDevicePropertiesNoThrow) {
    ov::Core ie = createCoreWithTemplate();
    OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, ov::test::utils::DEVICE_HETERO,
                                        ov::device::priorities(target_device),
                                        ov::device::properties(target_device, ov::enable_profiling(true))));
}

TEST_P(OVClassLoadNetworkTestVpux, LoadNetworkHETEROWithDeviceIDNoThrow) {
    ov::Core ie = createCoreWithTemplate();

    if (supportsDeviceID(ie, target_device)) {
        auto deviceIDs = ie.get_property(target_device, ov::available_devices);
        if (deviceIDs.empty())
            GTEST_FAIL();
        std::string heteroDevice = ov::test::utils::DEVICE_HETERO + std::string(":") + target_device + "." +
                                   deviceIDs[0] + "," + target_device;
        OV_ASSERT_NO_THROW(ie.compile_model(actualNetwork, heteroDevice));
    } else {
        GTEST_FAIL() << "Device does not support DeviceID property" << std::endl;
    }
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

TEST_P(OVClassBasicTestPVpux, smoke_registerPluginsLibrariesUnicodePath) {
    ov::Core core = createCoreWithTemplate();

    const std::vector<std::string> libs = {pluginName, std::string("npu_level_zero_backend") + OV_BUILD_POSTFIX};

    for (std::size_t testIndex = 0; testIndex < ov::test::utils::test_unicode_postfix_vector.size(); testIndex++) {
        GTEST_COUT << testIndex;
        std::string unicode_target_device = target_device + "_UNICODE_" + std::to_string(testIndex);
        std::wstring postfix = ov::test::utils::test_unicode_postfix_vector[testIndex];
        std::wstring unicode_path = ov::test::utils::stringToWString(ov::util::get_ov_lib_path() + "/") + postfix;
        try {
#ifndef _WIN32
            std::filesystem::create_directory(ov::util::wstring_to_string(unicode_path));
#else
            std::filesystem::create_directory(unicode_path);
#endif
            std::string pluginNamePath =
                    ov::util::make_plugin_library_name(ov::util::wstring_to_string(unicode_path), pluginName);

            for (auto&& lib : libs) {
                auto&& libPath = ov::test::utils::stringToWString(
                        ov::util::make_plugin_library_name(ov::util::get_ov_lib_path(), lib));
                auto&& libPathNew = ov::test::utils::stringToWString(
                        ov::util::make_plugin_library_name(::ov::util::wstring_to_string(unicode_path), lib));
                bool is_copy_successfully = ov::test::utils::copyFile(libPath, libPathNew);
                if (!is_copy_successfully) {
                    FAIL() << "Unable to copy from '" << libPath << "' to '" << libPathNew << "'";
                }
            }

            OV_ASSERT_NO_THROW(core.register_plugin(pluginNamePath, unicode_target_device));
            OV_ASSERT_NO_THROW(core.get_versions(unicode_target_device));
            auto devices = core.get_available_devices();
            if (std::find_if(devices.begin(), devices.end(), [&unicode_target_device](std::string device) {
                    return device.find(unicode_target_device) != std::string::npos;
                }) == devices.end()) {
                FAIL() << unicode_target_device << " was not found within registered plugins.";
            }
            core.unload_plugin(unicode_target_device);
        } catch (const ov::Exception& e_next) {
            FAIL() << e_next.what();
        }
    }
}
#endif

}  // namespace behavior
}  // namespace test
}  // namespace ov
