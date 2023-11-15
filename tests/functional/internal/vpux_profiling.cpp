//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <functional_test_utils/skip_tests_config.hpp>

#include "test_model/profiling_test_base.hpp"
#include "vpux_private_config.hpp"
#include "vpux_private_properties.hpp"

using ov::intel_vpux::CompilerType;

class ProfilingTest_VPU3700 : public ProfilingTestBase, public testing::WithParamInterface<RunProfilingParams> {};
class ProfilingTest_VPU3720 : public ProfilingTestBase, public testing::WithParamInterface<RunProfilingParams> {};

TEST_P(ProfilingTest_VPU3700, RunsProfiling) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    SKIP_ON("EMULATOR", "Not supported");

    const auto& profilingParams = GetParam();

    std::set<std::string> layerExecTypes = {"SW", "DMA"};
    // Empirical value from platform 3700 is 135 microseconds.
    auto execTimeRange = std::make_pair(1ll, 2000ll);
    ProfilingTaskCheckParams checkParams{layerExecTypes, execTimeRange};
    runTest(profilingParams, checkParams);
}

TEST_P(ProfilingTest_VPU3720, RunsProfiling) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    SKIP_ON("EMULATOR", "Not supported");

    const auto& profilingParams = GetParam();

    std::set<std::string> layerExecTypes = {"SW", "DPU", "DMA"};
    // Empirical values on 3720 range from ~5us to ~2500us depending whether it's realTime
    // or cpu time duration of a profiling task.
    auto execTimeRange = std::make_pair(1ll, 5000ll);  // extending the margin due to intrinsic performance variability
    ProfilingTaskCheckParams checkParams{layerExecTypes, execTimeRange};
    runTest(profilingParams, checkParams);
}

// VPU3700
INSTANTIATE_TEST_SUITE_P(precommit_profilingDisabled, ProfilingTest_VPU3700,
                         testing::Values(RunProfilingParams{"conv", CompilerType::MLIR, false}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingMatchedName, ProfilingTest_VPU3700,
                         testing::Values(RunProfilingParams{"Result", CompilerType::MLIR, true}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingNonMatchedName, ProfilingTest_VPU3700,
                         testing::Values(RunProfilingParams{"conv", CompilerType::MLIR, true}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingDisabled_drv, ProfilingTest_VPU3700,
                         testing::Values(RunProfilingParams{"conv", CompilerType::DRIVER, false}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingMatchedName_drv, ProfilingTest_VPU3700,
                         testing::Values(RunProfilingParams{"Result", CompilerType::DRIVER, true}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingNonMatchedName_drv, ProfilingTest_VPU3700,
                         testing::Values(RunProfilingParams{"conv", CompilerType::DRIVER, true}));

// VPU3720
INSTANTIATE_TEST_SUITE_P(precommit_profilingDisabled, ProfilingTest_VPU3720,
                         testing::Values(RunProfilingParams{"conv", CompilerType::MLIR, false}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingMatchedName, ProfilingTest_VPU3720,
                         testing::Values(RunProfilingParams{"Result", CompilerType::MLIR, true}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingNonMatchedName, ProfilingTest_VPU3720,
                         testing::Values(RunProfilingParams{"conv", CompilerType::MLIR, true}));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_precommit_profilingDisabled_drv, ProfilingTest_VPU3720,
                         testing::Values(RunProfilingParams{"conv", CompilerType::DRIVER, false}));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_precommit_profilingMatchedName_drv, ProfilingTest_VPU3720,
                         testing::Values(RunProfilingParams{"Result", CompilerType::DRIVER, true}));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_precommit_profilingNonMatchedName_drv, ProfilingTest_VPU3720,
                         testing::Values(RunProfilingParams{"conv", CompilerType::DRIVER, true}));
