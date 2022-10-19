//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include "test_model/kmb_test_base.hpp"
#include "vpux_private_config.hpp"

enum class CompilerType {DRIVER, MCM, MLIR};

struct RunProfilingParams final {
    const std::string outputName;
    CompilerType compiler;
    bool profiling;
};

class KmbProfilingTest : public KmbLayerTestBase, public testing::WithParamInterface<RunProfilingParams> {};


TEST_P(KmbProfilingTest, RunsProfiling) {
    SKIP_ON("HDDL2", "EMULATOR", "Not supported");
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
    case CompilerType::MCM:
        netConfig[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MCM);
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


    if (RUN_COMPILER)
    {
        TestNetwork testNet;
        testNet
            .setUserInput("input", userInDesc.getPrecision(), userInDesc.getLayout())
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

    if (RUN_INFER)
    {
        ExecutableNetwork exeNet = KmbTestBase::importNetwork(netConfig);
        auto inferRequest = exeNet.CreateInferRequest();

        inferRequest.Infer();

        if (profiling) {
            std::map<std::string, InferenceEngineProfileInfo> perfMap = inferRequest.GetPerformanceCounts();
            ASSERT_NE(perfMap.size(), 0);
        }

    /* This is the example of extracting per layer info (reference for the future tests expansion)
        std::vector<std::pair<std::string, InferenceEngineProfileInfo>> perfVec(perfMap.begin(), perfMap.end());
        std::sort(perfVec.begin(), perfVec.end(),
        [=](const std::pair<std::string, InferenceEngineProfileInfo>& pair1,
            const std::pair<std::string, InferenceEngineProfileInfo>& pair2) -> bool {
            return pair1.second.execution_index < pair2.second.execution_index;
        });

        for (auto& it : perfVec) {
            std::string layerName = it.first;
            InferenceEngineProfileInfo info = it.second;
            std::cout << layerName << " : " << info.realTime_uSec << std::endl;
        }
    */
    }
}

#if defined(_WIN32) || defined(_WIN64)
// Compiler is located in driver by default
INSTANTIATE_TEST_SUITE_P(precommit_profilingMatchedName, KmbProfilingTest,
                         testing::Values(RunProfilingParams{"Result", CompilerType::DRIVER, true}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingNonMatchedName, KmbProfilingTest,
                         testing::Values(RunProfilingParams{"conv", CompilerType::DRIVER, true}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingDisabled, KmbProfilingTest,
                         testing::Values(RunProfilingParams{"conv", CompilerType::DRIVER, false}));
#else
INSTANTIATE_TEST_SUITE_P(precommit_profilingMatchedName, KmbProfilingTest,
                         testing::Values(RunProfilingParams{"Result", CompilerType::MLIR, true}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingNonMatchedName, KmbProfilingTest,
                         testing::Values(RunProfilingParams{"conv", CompilerType::MLIR, true}));

INSTANTIATE_TEST_SUITE_P(precommit_profilingDisabled, KmbProfilingTest,
                         testing::Values(RunProfilingParams{"conv", CompilerType::MLIR, false}));
#endif
