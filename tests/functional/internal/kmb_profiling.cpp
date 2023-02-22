//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <functional_test_utils/skip_tests_config.hpp>
#include "test_model/kmb_test_base.hpp"
#include "vpux_private_config.hpp"

enum class CompilerType { DRIVER, MLIR };

std::string toStr(CompilerType ct) {
    if (ct == CompilerType::MLIR)
        return "MLIR";
    if (ct == CompilerType::DRIVER)
        return "DRIVER";
    return "UNKNOWN";
}

struct RunProfilingParams final {
    const std::string outputName;
    CompilerType compiler;
    bool profiling;
};

std::ostream& operator<<(std::ostream& os, const RunProfilingParams& p) {
    vpux::printTo(os, "outputName: {0}, compiler: {1}, profiling: {2}", p.outputName, toStr(p.compiler), p.profiling);
    return os;
}

class KmbProfilingTest : public KmbLayerTestBase, public testing::WithParamInterface<RunProfilingParams> {};

TEST_P(KmbProfilingTest, RunsProfiling) {
    SKIP_ON("HDDL2", "EMULATOR", "Not supported");
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    const auto& p = GetParam();
    const std::string outputName = p.outputName;
    const bool profiling = p.profiling;

    const SizeVector inDims = {1, 3, 64, 64};
    const TensorDesc userInDesc = TensorDesc(Precision::U8, inDims, Layout::NHWC);
    const TensorDesc userOutDesc = TensorDesc(Precision::FP16, Layout::NHWC);
    const auto scaleDesc = TensorDesc(Precision::FP32, inDims, Layout::NHWC);
    const Precision netPresicion = Precision::FP32;
    std::map<std::string, std::string> netConfig;

    if (profiling) {
        netConfig[CONFIG_KEY(PERF_COUNT)] = CONFIG_VALUE(YES);
    }
    switch (p.compiler) {
    case CompilerType::DRIVER:
        netConfig[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(DRIVER);
        break;
    case CompilerType::MLIR:
        netConfig[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);
        break;
    }

    registerBlobGenerator("input", userInDesc, [&](const TensorDesc& desc) {
        return vpux::makeSplatBlob(desc, 1.0f);
    });
    registerBlobGenerator("scale", scaleDesc, [&](const TensorDesc& desc) {
        return vpux::makeSplatBlob(desc, 1.f);
    });

    if (RUN_COMPILER) {
        TestNetwork testNet;
        testNet.setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
                .addNetInput("input", userInDesc.getDims(), netPresicion)
                .addLayer<PowerLayerDef>(outputName)
                .input1("input")
                .input2(getBlobByName("scale"))
                .build()
                .addNetOutput(PortInfo(outputName))
                .setUserOutput(PortInfo(outputName), userOutDesc.getPrecision(), userOutDesc.getLayout())
                .setCompileConfig(netConfig)
                .finalize();

        ExecutableNetwork exeNet = getExecNetwork(testNet);
        KmbTestBase::exportNetwork(exeNet);
    }

    if (RUN_INFER) {
        ExecutableNetwork exeNet = KmbTestBase::importNetwork(netConfig);
        auto inferRequest = exeNet.CreateInferRequest();

        inferRequest.Infer();

        if (profiling) {
            auto inTheRange = [](long long val, long long min, long long max) {
                return val > min && val < max;
            };
            std::map<std::string, InferenceEngineProfileInfo> perfMap = inferRequest.GetPerformanceCounts();
            ASSERT_NE(perfMap.size(), 0);
            const long long expectedMinTimeUs = 1ll;
            const long long expectedMaxTimeUs = 2000ll;  // increased threshold to account for delay due to DMA
            for (const auto& layer : perfMap) {
                const auto cpu = layer.second.cpu_uSec;
                const auto real = layer.second.realTime_uSec;
                // Empirical value from platform 3700 is 135 microseconds.
                ASSERT_PRED3(inTheRange, cpu, expectedMinTimeUs, expectedMaxTimeUs)
                        << "CPU duration " << cpu << " is out of range.";
                ASSERT_PRED3(inTheRange, real, expectedMinTimeUs, expectedMaxTimeUs)
                        << "Real time duration " << real << " is out of range.";
                ASSERT_TRUE(std::string(layer.second.exec_type) == "SW" ||
                            std::string(layer.second.exec_type) == "DMA");
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(precommit_profilingDisabled, KmbProfilingTest,
                         testing::Values(RunProfilingParams{"conv", CompilerType::MLIR, false}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingMatchedName, KmbProfilingTest,
                         testing::Values(RunProfilingParams{"Result", CompilerType::MLIR, true}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingNonMatchedName, KmbProfilingTest,
                         testing::Values(RunProfilingParams{"conv", CompilerType::MLIR, true}));

// TODO: E#61865 turn DRIVER tests on once driver is ready
#if defined(_WIN32) || defined(_WIN64)
// Compiler is located in driver by default
INSTANTIATE_TEST_SUITE_P(precommit_profilingDisabled_drv, KmbProfilingTest,
                         testing::Values(RunProfilingParams{"conv", CompilerType::DRIVER, false}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingMatchedName_drv, KmbProfilingTest,
                         testing::Values(RunProfilingParams{"Result", CompilerType::DRIVER, true}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingNonMatchedName_drv, KmbProfilingTest,
                         testing::Values(RunProfilingParams{"conv", CompilerType::DRIVER, true}));
#endif
