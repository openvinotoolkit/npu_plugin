//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "profiling_test_base.hpp"
#include "vpux_private_config.hpp"
#include "vpux_private_properties.hpp"

using ov::intel_vpux::CompilerType;

//
// ProfilingTestBase
//

void ProfilingTestBase::runTest(RunProfilingParams profilingRunParams, ProfilingTaskCheckParams profilingCheckParams) {
    const std::string outputName = profilingRunParams.outputName;
    const bool profiling = profilingRunParams.profiling;
    auto allowedTaskExecTypes = profilingCheckParams.allowedTaskExecTypes;
    auto execTimeRange = profilingCheckParams.execTimeRange;

    const SizeVector inDims = {1, 3, 64, 64};
    const TensorDesc userInDesc = TensorDesc(Precision::U8, inDims, Layout::NHWC);
    const TensorDesc userOutDesc = TensorDesc(Precision::FP16, Layout::NHWC);
    const auto scaleDesc = TensorDesc(Precision::FP32, inDims, Layout::NHWC);
    const Precision netPresicion = Precision::FP32;
    std::map<std::string, std::string> netConfig;

    if (profiling) {
        netConfig[CONFIG_KEY(PERF_COUNT)] = CONFIG_VALUE(YES);
    }
    switch (profilingRunParams.compiler) {
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
            auto inAllowedExecTypes = [&](std::string execType) {
                return allowedTaskExecTypes.find(execType) != allowedTaskExecTypes.end();
            };
            std::map<std::string, InferenceEngineProfileInfo> perfMap = inferRequest.GetPerformanceCounts();
            ASSERT_NE(perfMap.size(), 0);
            const long long expectedMinTimeUs = execTimeRange.first;
            const long long expectedMaxTimeUs = execTimeRange.second;
            for (const auto& layer : perfMap) {
                const auto cpu = layer.second.cpu_uSec;
                const auto real = layer.second.realTime_uSec;
                ASSERT_PRED3(inTheRange, cpu, expectedMinTimeUs, expectedMaxTimeUs)
                        << "CPU duration " << cpu << " is out of range.";
                ASSERT_PRED3(inTheRange, real, expectedMinTimeUs, expectedMaxTimeUs)
                        << "Real time duration " << real << " is out of range.";
                ASSERT_PRED1(inAllowedExecTypes, layer.second.exec_type);
            }
        }
    }
}
