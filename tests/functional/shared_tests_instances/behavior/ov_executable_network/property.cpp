// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/plugin_cache.hpp"
#include "common/functions.h"
#include "behavior/ov_executable_network/properties.hpp"

#include <vpux/properties.hpp>

using namespace ov::test::behavior;

namespace {

const char* CSRAM_TEST_VALUE = "0";

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

using VPUXClassExecutableNetworkTrySetImmutableTest  = VPUXClassExecutableNetworkGetPropertiesTest;


TEST_P(VPUXClassExecutableNetworkGetPropertiesTest, CheckPropertyIsSupportedAndGet) {
    std::vector<ov::PropertyName> properties;

    ov::CompiledModel exeNetwork = ie.compile_model(model, deviceName);
    ASSERT_NO_THROW(properties = exeNetwork.get_property(ov::supported_properties));

    auto it = find(properties.cbegin(), properties.cend(), configKey);
    ASSERT_TRUE(it != properties.cend());
    ASSERT_TRUE(it->is_mutable());
}

INSTANTIATE_TEST_SUITE_P(
    smoke_VPUXClassExecutableNetworkGetPropertiesTest,
    VPUXClassExecutableNetworkGetPropertiesTest,
    ::testing::Combine(
    ::testing::Values(getDeviceName()),
    ::testing::Values(std::make_pair(ov::intel_vpux::csram_size.name(), ov::intel_vpux::csram_size(CSRAM_TEST_VALUE)))));

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