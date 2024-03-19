//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "profiling_test_base.hpp"
#include "test_add_def.hpp"
#include "vpux_private_properties.hpp"

using ov::intel_vpux::CompilerType;

//
// ProfilingTestBase
//

void ProfilingTestBase::runTest(RunProfilingParams profilingRunParams, ProfilingTaskCheckParams profilingCheckParams) {
    const bool profiling = profilingRunParams.profiling;
    const auto nTiles = profilingRunParams.nTiles;
    auto allowedTaskExecTypes = profilingCheckParams.allowedTaskExecTypes;
    auto expectedExecTimes = profilingCheckParams.layerExecTimes;

    const SizeVector inDims = {1, 3, 128, 128};
    const TensorDesc userInDesc = TensorDesc(
            Precision::FP32, inDims, Layout::NCHW);  // using compatible layout in order to avoid transpose operations
    const TensorDesc userOutDesc =
            TensorDesc(Precision::FP32, Layout::NCHW);  // using FP32 in order to avoid convert operations
    const auto scaleDesc = TensorDesc(Precision::FP32, inDims, Layout::NCHW);
    const Precision netPresicion = Precision::FP32;
    std::map<std::string, std::string> netConfig;

    if (profiling) {
        netConfig[CONFIG_KEY(PERF_COUNT)] = CONFIG_VALUE(YES);
    }
    switch (profilingRunParams.compiler) {
    case CompilerType::DRIVER:
        netConfig[ov::intel_vpux::compiler_type.name()] = "DRIVER";
        break;
    case CompilerType::MLIR:
        netConfig[ov::intel_vpux::compiler_type.name()] = "MLIR";
        break;
    }

    if (nTiles.has_value()) {
        netConfig[ov::intel_vpux::dpu_groups.name()] = std::to_string(nTiles.value());
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
                .addLayer<PowerLayerDef>("Power")
                .input1("input")
                .input2(getBlobByName("scale"))
                .build()
                .addLayer<AddLayerDef>("Add")
                .input1("input")
                .input2("Power")
                .build()
                .addLayer<SoftmaxLayerDef>("Softmax", 1)
                .input("Add")
                .build()
                .addNetOutput(PortInfo("Softmax"))
                .setUserOutput(PortInfo("Softmax"), userOutDesc.getPrecision(), userOutDesc.getLayout())
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
                return val >= min && val <= max;
            };
            auto inAllowedExecTypes = [&](std::string execType) {
                return allowedTaskExecTypes.find(execType) != allowedTaskExecTypes.end();
            };
            auto lowerCase = [](std::string str) {
                std::transform(str.begin(), str.end(), str.begin(), ::tolower);
                return str;
            };

            std::map<std::string, InferenceEngineProfileInfo> perfMap = inferRequest.GetPerformanceCounts();
            ASSERT_NE(perfMap.size(), 0);

            // assure correct exec types
            for (const auto& layer : perfMap) {
                ASSERT_PRED1(inAllowedExecTypes, layer.second.exec_type);
            }

            // assure correct layer names and types are present in the performance counts map
            for (const auto& [expectedLayerName, layerExecTimes] : expectedExecTimes) {
                const std::string expectedLayerType = expectedLayerName;  // the two are set to coincide in the model
                ASSERT_TRUE(perfMap.find(expectedLayerName) != perfMap.end())
                        << "Could not find the expected layer name: " << expectedLayerName
                        << " in the performance map.";
                const auto testLayer = perfMap[expectedLayerName];
                ASSERT_TRUE(lowerCase(testLayer.layer_type) == lowerCase(expectedLayerType))
                        << "Layer " << expectedLayerName << " has unexpected type (" << testLayer.layer_type
                        << "). Expected type: "
                        << expectedLayerType;  // ignore possible cross-platform layer type naming differences
                                               // from ngraph standards (eg. SoftMax vs Softmax)

                const auto cpuTime = testLayer.cpu_uSec;
                const auto realTime = testLayer.realTime_uSec;
                auto [expectedMinCpuTimeUs, expectedMaxCpuTimeUs, expectedMaxRealTimeUs] = layerExecTimes;
                ASSERT_PRED3(inTheRange, cpuTime, expectedMinCpuTimeUs, expectedMaxCpuTimeUs)
                        << "CPU time " << cpuTime << "us is out of range.";
                ASSERT_PRED3(inTheRange, realTime, expectedMinCpuTimeUs, expectedMaxRealTimeUs)
                        << "real time " << realTime << "us is out of range.";
                // real time can be smaller than the cpu time depending on number of tiles. We use the
                // expectedMinCpuTimeUs bound, which accounts for the dispersion due to the number of tiles.
            }
        }
    }
}
