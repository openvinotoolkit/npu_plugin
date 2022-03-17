// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/plugin_cache.hpp"
#include "common/functions.h"
#include "behavior/ov_executable_network/properties.hpp"
#include "vpux/al/config/common.hpp"

#include <vpux/properties.hpp>

#include <vector>

using namespace ov::test::behavior;

namespace {

std::vector<std::pair<std::string, ov::Any>> expected_supported_properties_exe_network = {
    {ov::intel_vpux::csram_size.name(), ov::Any(0)},
    {ov::intel_vpux::executor_streams.name(), ov::Any(2)},
    {ov::intel_vpux::graph_color_format.name(), ov::Any(ov::intel_vpux::ColorFormat::RGB)},
    {ov::intel_vpux::inference_shaves.name(), ov::Any(2)},
    {ov::intel_vpux::inference_timeout.name(), ov::Any(2000)},
    {ov::intel_vpux::preprocessing_lpi.name(), ov::Any(4)},
    {ov::intel_vpux::preprocessing_pipes.name(), ov::Any(2)},
    {ov::intel_vpux::preprocessing_shaves.name(), ov::Any(2)},
    {ov::intel_vpux::print_profiling.name(), ov::Any(ov::intel_vpux::ProfilingOutputTypeArg::JSON)},
    {ov::intel_vpux::profiling_output_file.name(), ov::Any("some/file")},
    {ov::intel_vpux::use_m2i.name(), ov::Any(true)},
    {ov::intel_vpux::use_shave_only_m2i.name(), ov::Any(true)},
    {ov::intel_vpux::use_sipp.name(), ov::Any(false)},
    {ov::intel_vpux::vpux_platform.name(), ov::Any(ov::intel_vpux::VPUXPlatform::EMULATOR)},
    {ov::hint::model_priority.name(), ov::Any(ov::hint::Priority::HIGH)}
};

std::vector<ov::AnyMap> expected_supported_properties_plugin = {
    {ov::intel_vpux::compilation_descriptor("some_arbitrary")},
    {ov::intel_vpux::compilation_descriptor_path("some/path/descriptor")},
    {ov::intel_vpux::compilation_pass_ban_list("group, pass")},
    {ov::intel_vpux::concat_scales_alignment("NO")},
    {ov::intel_vpux::csram_size("0")},
    {ov::intel_vpux::custom_layers("some/xml/file.xml")},
    {ov::intel_vpux::eltwise_scales_alignment("NO")},
    {ov::intel_vpux::inference_shaves(2)},
    {ov::intel_vpux::remove_permute_noop("NO")},
    {ov::intel_vpux::scale_fuse_input("NO")},
    {ov::intel_vpux::target_descriptor("some_descriptor")},
    {ov::intel_vpux::target_descriptor_path("some/target/path")},
    {ov::intel_vpux::weights_zero_points_alignment("NO")}
};

std::string getDeviceName() {
    auto* env_val = std::getenv("IE_KMB_TESTS_DEVICE_NAME");
    return (env_val != nullptr) ? std::getenv("IE_KMB_TESTS_DEVICE_NAME") : "VPUX.3700";
}

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
};

class VPUXClassPluginPropertiesTest:
        public OVCompiledModelPropertiesBase,
        public ::testing::WithParamInterface<std::tuple<std::string, ov::AnyMap>> {
protected:
    std::string deviceName;
    ov::AnyMap configMap;
    ov::Core ie;
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        OVCompiledModelPropertiesBase::SetUp();
        std::tie(deviceName, configMap) = GetParam();
    }
};

using VPUXClassExecutableNetworkTrySetImmutableTest = VPUXClassExecutableNetworkGetPropertiesTest;
using VPUXClassExecutableNetworkTryGetDirectTest = VPUXClassExecutableNetworkGetPropertiesTest;
using VPUXClassPluginTryGetSetPropertyTest = VPUXClassPluginPropertiesTest;

TEST_P(VPUXClassExecutableNetworkGetPropertiesTest, CheckPropertyIsSupportedAndGet) {
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel exeNetwork = ie.compile_model(model, deviceName);
    ASSERT_NO_THROW(properties = exeNetwork.get_property(ov::supported_properties));

    auto it = find(properties.cbegin(), properties.cend(), configKey);
    ASSERT_TRUE(it != properties.cend());
    ASSERT_TRUE(!it->is_mutable());

    ASSERT_THROW(exeNetwork.set_property({{configKey, configValue.as<std::string>()}}), ov::Exception);

    ov::Any retreived_value;
    ASSERT_NO_THROW(retreived_value = exeNetwork.get_property(configKey));
}

INSTANTIATE_TEST_SUITE_P(
    smoke_VPUXClassExecutableNetworkGetPropertiesTest,
    VPUXClassExecutableNetworkGetPropertiesTest,
    ::testing::Combine(
    ::testing::Values(getDeviceName()),
    ::testing::ValuesIn(expected_supported_properties_exe_network)));

TEST_P(VPUXClassPluginTryGetSetPropertyTest, GetSetPropertyPlugin) {
    ASSERT_EQ(configMap.size(), 1); 
    std::vector<ov::PropertyName> properties;

    ASSERT_NO_THROW(properties = ie.get_property(deviceName, ov::supported_properties));

    auto it = find(properties.cbegin(), properties.cend(), configMap.begin()->first);
    ASSERT_TRUE(it != properties.cend());
    ASSERT_TRUE(it->is_mutable());

    ASSERT_NO_THROW(ie.set_property(deviceName, configMap));

    ov::Any retrieved_value;
    ASSERT_NO_THROW(retrieved_value = ie.get_property(deviceName, configMap.begin()->first));

    ASSERT_EQ(retrieved_value.as<std::string>(), configMap.begin()->second.as<std::string>());
}

INSTANTIATE_TEST_SUITE_P(
    smoke_VPUXClassPluginTryGetSetPropertyTest,
    VPUXClassPluginTryGetSetPropertyTest,
    ::testing::Combine(
    ::testing::Values(getDeviceName()),
    ::testing::ValuesIn(expected_supported_properties_plugin)));

TEST_P(VPUXClassExecutableNetworkTrySetImmutableTest, TryToSetImmutableProperty) {
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel exeNetwork = ie.compile_model(model, deviceName);
    ASSERT_NO_THROW(properties = exeNetwork.get_property(ov::supported_properties));

    auto it = find(properties.cbegin(), properties.cend(), configKey);
    ASSERT_TRUE(it != properties.cend());
    ASSERT_FALSE(it->is_mutable());

    ASSERT_THROW(exeNetwork.set_property({{configKey, configValue}}), ov::Exception);
}

INSTANTIATE_TEST_SUITE_P(
    smoke_VPUXClassExecutableNetworkTrySetImmutableTest,
    VPUXClassExecutableNetworkTrySetImmutableTest,
    ::testing::Combine(
    ::testing::Values(getDeviceName()),
    ::testing::Values(std::make_pair(ov::optimal_number_of_infer_requests.name(), ov::Any(2)))));

}