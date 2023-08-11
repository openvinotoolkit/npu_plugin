//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_executable_network/properties.hpp"
#include "common/functions.h"
#include "functional_test_utils/plugin_cache.hpp"
#include "vpux/al/config/common.hpp"

#include <openvino/runtime/device_id_parser.hpp>
#include <vpux/properties.hpp>

#include <vector>

using namespace ov::test::behavior;

namespace {

std::string getDeviceName() {
    auto* env_val = std::getenv("IE_KMB_TESTS_DEVICE_NAME");
    return (env_val != nullptr) ? env_val : "VPUX.3700";
}

std::string getDeviceNameID(const std::string& str) {
    ov::DeviceIDParser parser = ov::DeviceIDParser(str);
    return parser.get_device_id();
}

std::string getDeviceNameTestCase(const std::string& str) {
    ov::DeviceIDParser parser = ov::DeviceIDParser(str);
    return parser.get_device_name().substr(0, parser.get_device_name().size() - 1) + parser.get_device_id();
}

std::vector<std::pair<std::string, ov::Any>> exe_network_supported_properties = {
        {ov::intel_vpux::print_profiling.name(), ov::Any(ov::intel_vpux::ProfilingOutputTypeArg::JSON)},
        {ov::intel_vpux::profiling_output_file.name(), ov::Any("some/file")},
        {ov::intel_vpux::vpux_platform.name(), ov::Any(ov::intel_vpux::VPUXPlatform::EMULATOR)},
        {ov::intel_vpux::ddr_heap_size_mb.name(), ov::Any(500)},
        {ov::hint::model_priority.name(), ov::Any(ov::hint::Priority::HIGH)},
        {ov::hint::num_requests.name(), ov::Any(8)},
        {ov::hint::performance_mode.name(), ov::Any(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::enable_profiling.name(), ov::Any(true)},
        {ov::device::id.name(), ov::Any(getDeviceNameID(getDeviceName()))},
        {ov::intel_vpux::use_elf_compiler_backend.name(), ov::Any(ov::intel_vpux::ElfCompilerBackend::YES)},
        {ov::intel_vpux::create_executor.name(), ov::Any(0)}};

std::vector<std::pair<std::string, ov::Any>> exe_network_immutable_properties = {
        {std::make_pair(ov::optimal_number_of_infer_requests.name(), ov::Any(2))},
        {std::make_pair(ov::supported_properties.name(), ov::Any("deadbeef"))},
        {std::make_pair(ov::model_name.name(), ov::Any("deadbeef"))}};

class VPUXClassExecutableNetworkGetPropertiesTest :
        public OVCompiledModelPropertiesBase,
        public ::testing::WithParamInterface<std::tuple<std::string, std::pair<std::string, ov::Any>>> {
protected:
    std::string deviceName;
    std::string configKey;
    ov::Any configValue;
    ov::Core ie;

public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        OVCompiledModelPropertiesBase::SetUp();
        deviceName = std::get<0>(GetParam());
        std::tie(configKey, configValue) = std::get<1>(GetParam());

        model = ngraph::builder::subgraph::makeConvPoolRelu();
    }
    static std::string getTestCaseName(
            testing::TestParamInfo<std::tuple<std::string, std::pair<std::string, ov::Any>>> obj) {
        std::string targetDevice;
        std::pair<std::string, ov::Any> configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        std::ostringstream result;
        result << "targetDevice=" << getDeviceNameTestCase(targetDevice) << "_";
        result << "config=(" << configuration.first << "=" << configuration.second.as<std::string>() << ")";
        return result.str();
    }
};

using VPUXClassExecutableNetworkTestSuite1 = VPUXClassExecutableNetworkGetPropertiesTest;

TEST_P(VPUXClassExecutableNetworkTestSuite1, PropertyIsSupportedAndImmutableAndGet) {
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel exeNetwork = ie.compile_model(model, deviceName);
    ASSERT_NO_THROW(properties = exeNetwork.get_property(ov::supported_properties));

    auto it = find(properties.cbegin(), properties.cend(), configKey);
    ASSERT_TRUE(it != properties.cend());
    ASSERT_FALSE(it->is_mutable());

    ASSERT_NO_THROW(exeNetwork.get_property(configKey));
}

INSTANTIATE_TEST_SUITE_P(smoke_VPUXClassExecutableNetworkGetPropertiesTest, VPUXClassExecutableNetworkTestSuite1,
                         ::testing::Combine(::testing::Values(getDeviceName()),
                                            ::testing::ValuesIn(exe_network_supported_properties)),
                         VPUXClassExecutableNetworkGetPropertiesTest::getTestCaseName);

using VPUXClassExecutableNetworkTestSuite2 = VPUXClassExecutableNetworkGetPropertiesTest;

TEST_P(VPUXClassExecutableNetworkTestSuite2, PropertyIsSupportedAndImmutableAndCanNotSet) {
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel exeNetwork = ie.compile_model(model, deviceName);
    ASSERT_NO_THROW(properties = exeNetwork.get_property(ov::supported_properties));

    auto it = find(properties.cbegin(), properties.cend(), configKey);
    ASSERT_TRUE(it != properties.cend());
    ASSERT_FALSE(it->is_mutable());

    ASSERT_THROW(exeNetwork.set_property({{configKey, configValue}}), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(smoke_VPUXClassExecutableNetworkTestSuite2, VPUXClassExecutableNetworkTestSuite2,
                         ::testing::Combine(::testing::Values(getDeviceName()),
                                            ::testing::ValuesIn(exe_network_immutable_properties)),
                         VPUXClassExecutableNetworkGetPropertiesTest::getTestCaseName);

}  // namespace

namespace {

std::vector<std::pair<std::string, ov::Any>> plugin_mutable_properties = {
        {ov::intel_vpux::force_host_precision_layout_conversion.name(), ov::Any(true)},
        {ov::hint::num_requests.name(), ov::Any(5)},
        {ov::intel_vpux::profiling_output_file.name(), ov::Any("some/file")},
        {ov::enable_profiling.name(), ov::Any(true)},
        {ov::hint::performance_mode.name(), ov::Any(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::log::level.name(), ov::Any(ov::log::Level::DEBUG)},
        {ov::device::id.name(), ov::Any(getDeviceNameID(getDeviceName()))},
        {ov::intel_vpux::print_profiling.name(), ov::Any(ov::intel_vpux::ProfilingOutputTypeArg::JSON)},
        {ov::intel_vpux::compiler_type.name(), ov::Any(ov::intel_vpux::CompilerType::MLIR)},
        {ov::intel_vpux::vpux_platform.name(), ov::Any(ov::intel_vpux::VPUXPlatform::AUTO_DETECT)},
        {ov::intel_vpux::compilation_mode.name(), ov::Any("DefaultHW")},
        {ov::intel_vpux::compilation_mode_params.name(), ov::Any("use-user-precision=false propagate-quant-dequant=0")},
        {ov::intel_vpux::dpu_groups.name(), ov::Any(2)},
        {ov::intel_vpux::use_elf_compiler_backend.name(), ov::Any(ov::intel_vpux::ElfCompilerBackend::YES)},
        {ov::intel_vpux::dma_engines.name(), ov::Any(1)}};

std::vector<std::pair<std::string, ov::Any>> plugin_immutable_properties = {
        {ov::device::uuid.name(), ov::Any("deadbeef")},
        {ov::supported_properties.name(), {ov::device::full_name.name()}},
        {ov::streams::num.name(), ov::Any(ov::streams::Num(4))},
        {ov::available_devices.name(), ov::Any(std::vector<std::string>{"deadbeef"})},
        {ov::device::capabilities.name(), ov::Any(std::vector<std::string>{"deadbeef"})},
        {ov::range_for_async_infer_requests.name(),
         ov::Any(std::tuple<unsigned int, unsigned int, unsigned int>{0, 10, 1})},
        {ov::range_for_streams.name(), ov::Any(std::tuple<unsigned int, unsigned int>{0, 10})},
        {ov::caching_properties.name(), ov::Any("deadbeef")}};

class VPUXClassPluginPropertiesTest :
        public OVCompiledModelPropertiesBase,
        public ::testing::WithParamInterface<std::tuple<std::string, std::pair<std::string, ov::Any>>> {
protected:
    std::string deviceName;
    std::string configKey;
    ov::Any configValue;
    ov::Core ie;

public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        OVCompiledModelPropertiesBase::SetUp();
        deviceName = std::get<0>(GetParam());
        std::tie(configKey, configValue) = std::get<1>(GetParam());
    }
    static std::string getTestCaseName(
            testing::TestParamInfo<std::tuple<std::string, std::pair<std::string, ov::Any>>> obj) {
        std::string targetDevice;
        std::pair<std::string, ov::Any> configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');
        std::ostringstream result;
        result << "targetDevice=" << getDeviceNameTestCase(targetDevice) << "_";
        result << "config=(" << configuration.first << "=" << configuration.second.as<std::string>() << ")";
        return result.str();
    }
};

using VPUXClassPluginPropertiesTestSuite1 = VPUXClassPluginPropertiesTest;

TEST_P(VPUXClassPluginPropertiesTestSuite1, CanSetGetMutableProperty) {
    std::vector<ov::PropertyName> properties;

    ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::supported_properties));

    auto it = find(properties.cbegin(), properties.cend(), configKey);
    ASSERT_TRUE(it != properties.cend());
    ASSERT_TRUE(it->is_mutable());

    ASSERT_NO_THROW(ie.set_property(deviceName, {{configKey, configValue}}));

    ov::Any retrieved_value;
    ASSERT_NO_THROW(retrieved_value = ie.get_property(deviceName, configKey));

    ASSERT_EQ(retrieved_value.as<std::string>(), configValue.as<std::string>());
}

INSTANTIATE_TEST_SUITE_P(smoke_VPUXClassPluginPropertiesTest, VPUXClassPluginPropertiesTestSuite1,
                         ::testing::Combine(::testing::Values(getDeviceName()),
                                            ::testing::ValuesIn(plugin_mutable_properties)),
                         VPUXClassPluginPropertiesTest::getTestCaseName);

using VPUXClassPluginPropertiesTestSuite2 = VPUXClassPluginPropertiesTest;

TEST_P(VPUXClassPluginPropertiesTestSuite2, CanNotSetImmutableProperty) {
    std::vector<ov::PropertyName> properties;

    ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::supported_properties));

    auto it = find(properties.cbegin(), properties.cend(), configKey);
    ASSERT_TRUE(it != properties.cend());
    ASSERT_FALSE(it->is_mutable());

    ov::Any orig_value;
    ASSERT_NO_THROW(orig_value = ie.get_property(deviceName, configKey));

    ASSERT_THROW(ie.set_property(deviceName, {{configKey, configValue}}), ov::Exception);

    ov::Any after_value;
    ASSERT_NO_THROW(after_value = ie.get_property(deviceName, configKey));

    ASSERT_EQ(orig_value.as<std::string>(), after_value.as<std::string>());
}

INSTANTIATE_TEST_SUITE_P(smoke_VPUXClassPluginPropertiesTest, VPUXClassPluginPropertiesTestSuite2,
                         ::testing::Combine(::testing::Values(getDeviceName()),
                                            ::testing::ValuesIn(plugin_immutable_properties)),
                         VPUXClassPluginPropertiesTest::getTestCaseName);

}  // namespace
