// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <base/ov_behavior_test_utils.hpp>
#include <ie_core.hpp>
#include <string>
#include <vector>
#include "common/functions.h"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "vpux/al/config/common.hpp"
#include "vpux_private_properties.hpp"

namespace {

class ElfConfigTests :
        public ov::test::behavior::OVPluginTestBase,
        public testing::WithParamInterface<std::tuple<std::string, ov::AnyMap>> {
public:
    void SetUp() override {
        std::tie(target_device, configuration) = GetParam();
        OVPluginTestBase::SetUp();
    }

    static std::string getTestCaseName(testing::TestParamInfo<std::tuple<std::string, ov::AnyMap>> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');
        std::ostringstream result;
        result << "targetDevice=" << targetDevice << "_";
        result << "targetPlatform=" << LayerTestsUtils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        if (!configuration.empty()) {
            using namespace ov::test::utils;
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }
        return result.str();
    }

protected:
    ov::AnyMap configuration;
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
};

TEST_P(ElfConfigTests, CompilationWithSpecificConfig) {
    if (getBackendName(*core) == "LEVEL0") {
        GTEST_SKIP() << "Skip due to failure on device";
    }
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        const auto& ov_model = buildSingleLayerSoftMaxNetwork();
        ASSERT_NO_THROW(auto compiled_model = core->compile_model(ov_model, target_device, configuration));
    }
}

const std::vector<ov::AnyMap> configs = {
        {{ov::intel_vpux::platform(ov::intel_vpux::VPUXPlatform::VPU3720)},
         {ov::intel_vpux::use_elf_compiler_backend(ov::intel_vpux::ElfCompilerBackend::NO)}}};

INSTANTIATE_TEST_SUITE_P(smoke_ELF, ElfConfigTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         ElfConfigTests::getTestCaseName);
}  // namespace
