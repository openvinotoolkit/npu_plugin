//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <functional_test_utils/skip_tests_config.hpp>

#include "test_model/profiling_test_base.hpp"
#include "vpux_private_properties.hpp"

using ov::intel_vpux::CompilerType;

class ProfilingTest_VPU3700 : public ProfilingTestBase, public testing::WithParamInterface<RunProfilingParams> {};
class ProfilingTest_VPU3720 : public ProfilingTestBase, public testing::WithParamInterface<RunProfilingParams> {};

TEST_P(ProfilingTest_VPU3700, RunsProfiling) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    const auto& profilingParams = GetParam();

    std::set<std::string> layerExecTypes = {"SW", "DMA"};
    // For given layer types the experimental values of cpu/real time typically fall into range [us]:
    //      Add (SW): [556, 558] / [766, 767]
    //      SoftMax (SW): [3502, 3505] / [3509, 3510]
    // Define expected minimal execution cpu time and expected maximal execution cpu and real times for each layer
    // (assuming about an order of magnitude margin around empirical values from the platform).
    const std::map<std::string, std::tuple<unsigned, unsigned, unsigned>> expectedExecTimes = {
            {"Add", std::make_tuple(50, 8000, 8000)},
            {"Softmax", std::make_tuple(300, 35000, 35000)}};
    ProfilingTaskCheckParams checkParams{layerExecTypes, expectedExecTimes};
    runTest(profilingParams, checkParams);
}

TEST_P(ProfilingTest_VPU3720, RunsProfiling) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    const auto& profilingParams = GetParam();

    std::set<std::string> layerExecTypes = {"SW", "DPU", "DMA"};
    // Empirical values depend on layer and timing measurement type
    // (realTime vs cpu time), NPU HW used, compiler/driver task scheduling strategy, and network structure and are
    // subject to intrinsic performance variations. For given layer types the experimentally observed values for
    // cpu/real time for MLIR case are [us]:
    //      Power (SW): [3012, 8949] / [3020, 7084]
    //      Add (DPU): [24, 39] / [2994, 5125]
    //      SoftMax (SW): [782, 2045] / [777, 1811]
    // for for DRIVER case:
    //      Power (SW): [3013, 5153] / [3021, 5167]
    //      Add (DPU): [24, 40] / [17, 28]
    //      SoftMax (SW): [786, 2044] / [781, 1810]
    // Define expected minimal execution cpu time and expected maximal execution cpu and real times for each layer
    // (assuming about an order of magnitude margin around empirical values from the platform).
    const std::map<std::string, std::tuple<unsigned, unsigned, unsigned>> expectedExecTimes = {
            {"Power", std::make_tuple(300, 90000, 90000)},
            {"Add", std::make_tuple(2, 400, 50000)},
            {"Softmax", std::make_tuple(80, 20000, 20000)}};

    ProfilingTaskCheckParams checkParams{layerExecTypes, expectedExecTimes};
    runTest(profilingParams, checkParams);
}

// VPU3700
INSTANTIATE_TEST_SUITE_P(precommit_profilingDisabled, ProfilingTest_VPU3700,
                         testing::Values(RunProfilingParams{CompilerType::MLIR, false}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingEnabled, ProfilingTest_VPU3700,
                         testing::Values(RunProfilingParams{CompilerType::MLIR, true}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingDisabled_drv, ProfilingTest_VPU3700,
                         testing::Values(RunProfilingParams{CompilerType::DRIVER, false}));

INSTANTIATE_TEST_SUITE_P(DISABLED_precommit_profilingEnabled_drv, ProfilingTest_VPU3700,
                         testing::Values(RunProfilingParams{CompilerType::DRIVER, true}));

// VPU3720
INSTANTIATE_TEST_SUITE_P(precommit_profilingDisabled, ProfilingTest_VPU3720,
                         testing::Values(RunProfilingParams{CompilerType::MLIR, false}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingEnabled, ProfilingTest_VPU3720,
                         testing::Values(RunProfilingParams{CompilerType::MLIR, true}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingDisabled_drv, ProfilingTest_VPU3720,
                         testing::Values(RunProfilingParams{CompilerType::DRIVER, false}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingEnabled_drv, ProfilingTest_VPU3720,
                         testing::Values(RunProfilingParams{CompilerType::DRIVER, true}));
