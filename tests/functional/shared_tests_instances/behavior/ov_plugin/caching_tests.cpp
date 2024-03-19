// Copyright (C) 2018-2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/ov_plugin/caching_tests.hpp"
#include "common/utils.hpp"
#include "common/vpu_test_env_cfg.hpp"
#include "vpux/al/config/common.hpp"

using namespace ov::test::behavior;

namespace {
static const std::vector<ngraph::element::Type> nightly_precisionsKeemBay = {
        ngraph::element::f32,
        // ngraph::element::f16,
        // ngraph::element::u8,
};

static const std::vector<ngraph::element::Type> smoke_precisionsKeemBay = {
        ngraph::element::f32,
};

static const std::vector<std::size_t> batchSizesKeemBay = {1};

static std::vector<ovModelWithName> smoke_functions() {
    auto funcs = CompileModelCacheTestBase::getStandardFunctions();
    if (funcs.size() > 2) {
        funcs.erase(funcs.begin() + 1, funcs.end());
    }
    return funcs;
}

static std::vector<ovModelWithName> keembay_functions() {
    auto funcs = CompileModelCacheTestBase::getStandardFunctions();

    std::vector<ovModelWithName>::iterator it = remove_if(funcs.begin(), funcs.end(), [](ovModelWithName func) {
        std::vector<std::string> bad_layers{"ReadConcatSplitAssign", "SimpleFunctionRelu",
                                            "2InputSubtract",        "MatMulBias",
                                            "TIwithLSTMcell1",       "KSOFunction"};
        return std::find(bad_layers.begin(), bad_layers.end(), std::get<1>(func)) != bad_layers.end();
    });

    funcs.erase(it, funcs.end());

    return funcs;
}

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_KeemBay, CompileModelCacheTestBase,
                         ::testing::Combine(::testing::ValuesIn(smoke_functions()),
                                            ::testing::ValuesIn(smoke_precisionsKeemBay),
                                            ::testing::ValuesIn(batchSizesKeemBay),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::Values(ov::AnyMap{})),
                         CompileModelCacheTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_CachingSupportCase_KeemBay, CompileModelCacheTestBase,
                         ::testing::Combine(::testing::ValuesIn(keembay_functions()),
                                            ::testing::ValuesIn(nightly_precisionsKeemBay),
                                            ::testing::ValuesIn(batchSizesKeemBay),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::Values(ov::AnyMap{})),
                         CompileModelCacheTestBase::getTestCaseName);

static std::string getTestCaseName(testing::TestParamInfo<compileModelLoadFromFileParams> obj) {
    std::string testCaseName = CompileModelLoadFromFileTestBase::getTestCaseName(obj);
    std::replace(testCaseName.begin(), testCaseName.end(), ':', '.');
    return testCaseName +
           "_targetPlatform=" + LayerTestsUtils::getTestsPlatformFromEnvironmentOr(ov::test::utils::DEVICE_NPU);
}

const std::vector<ov::AnyMap> LoadFromFileConfigs = {
        {ov::device::properties(ov::test::utils::DEVICE_NPU,
                                ov::intel_vpux::compiler_type(ov::intel_vpux::CompilerType::DRIVER)),
         ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::device::properties(ov::test::utils::DEVICE_NPU,
                                ov::intel_vpux::compiler_type(ov::intel_vpux::CompilerType::DRIVER)),
         ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}};
const std::vector<std::string> TestTargets = {
        ov::test::utils::DEVICE_AUTO,
        ov::test::utils::DEVICE_MULTI,
        ov::test::utils::DEVICE_BATCH,
};

INSTANTIATE_TEST_SUITE_P(smoke_Auto_CachingSupportCase_KeemBay, CompileModelLoadFromFileTestBase,
                         ::testing::Combine(::testing::ValuesIn(TestTargets), ::testing::ValuesIn(LoadFromFileConfigs)),
                         getTestCaseName);

const std::vector<ov::AnyMap> KEEMBAYLoadFromFileConfigs = {
        {ov::intel_vpux::compiler_type(ov::intel_vpux::CompilerType::DRIVER),
         ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)},
        {ov::intel_vpux::compiler_type(ov::intel_vpux::CompilerType::DRIVER),
         ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)},
        {ov::intel_vpux::compiler_type(ov::intel_vpux::CompilerType::DRIVER)},
};
INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_KeemBay, CompileModelLoadFromFileTestBase,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(KEEMBAYLoadFromFileConfigs)),
                         getTestCaseName);

}  // namespace
