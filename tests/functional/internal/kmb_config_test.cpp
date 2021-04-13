//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "test_model/kmb_test_base.hpp"

using ConfigMap = std::map<std::string, std::string>;
using ConfigTestParams = std::tuple<ConfigMap, ConfigMap, ConfigMap, ConfigMap, ConfigMap, ConfigMap>;

class KmbConfigTest : public KmbLayerTestBase, public testing::WithParamInterface<ConfigTestParams> {
public:
    void runTest(
        const ConfigMap& compileConfig,
        const ConfigMap& inferConfig);
};

void KmbConfigTest::runTest(
    const ConfigMap& compileConfig,
    const ConfigMap& inferConfig) {
    TestNetwork testNet;
    testNet
        .setUserInput("input", Precision::FP16, Layout::NCHW)
        .addNetInput("input", {1, 3, 512, 512}, Precision::FP32)
        .addLayer<SoftmaxLayerDef>("softmax", 1)
            .input("input")
            .build()
        .addNetOutput(PortInfo("softmax"))
        .setUserOutput(PortInfo("softmax"), Precision::FP16, Layout::NCHW)
        .finalize();

    testNet.setCompileConfig(compileConfig);
    ExecutableNetwork exeNet = KmbLayerTestBase::getExecNetwork(testNet);
    KmbTestBase::exportNetwork(exeNet);

    if (RUN_INFER) {
        ExecutableNetwork importedNet = KmbTestBase::importNetwork(inferConfig);
        constexpr bool printTime = false;
        const BlobMap inputs = {};
        KmbTestBase::runInfer(importedNet, inputs, printTime);
    }
}

TEST_P(KmbConfigTest, setConfig) {
    const auto& param = GetParam();
    const auto& compileConfig = std::get<0>(param);
    const auto& baseInferConfig = std::get<1>(param);
    const auto& colorFormatConfig = std::get<2>(param);
    const auto& preProcConfig = std::get<3>(param);
    const auto& preProcParamConfig = std::get<4>(param);
    const auto& inferShavesConfig = std::get<5>(param);
    ConfigMap inferConfig;
    inferConfig.insert(baseInferConfig.cbegin(), baseInferConfig.cend());
    inferConfig.insert(colorFormatConfig.cbegin(), colorFormatConfig.cend());
    inferConfig.insert(preProcConfig.cbegin(), preProcConfig.cend());
    inferConfig.insert(preProcParamConfig.cbegin(), preProcParamConfig.cend());
    inferConfig.insert(inferShavesConfig.cbegin(), inferShavesConfig.cend());

    runTest(compileConfig, inferConfig);
}

static const std::vector<ConfigMap> emptyConfigs = {
    { }
};

static const std::vector<ConfigMap> compileConfigs = {
    { }
};

static const std::vector<ConfigMap> baseInferConfigs = {
    { },
    { {"PERF_COUNT", "YES"} },
    { {"PERF_COUNT", "NO"} },
    //TODO Currently we can use any value
    { {"DEVICE_ID", "VPU-0"} },
    { {"VPUX_THROUGHPUT_STREAMS", "1" } },
    { {"VPUX_THROUGHPUT_STREAMS", "2"} },
    { {"VPUX_CSRAM_SIZE", "-1"} },
    { {"VPUX_CSRAM_SIZE", "0"} },
    { {"VPUX_CSRAM_SIZE", "2097152"} },
    { {"VPUX_EXECUTOR_STREAMS", "1"} },
    #if defined(__arm) || defined(__aarch64__)
    { {"VPUX_VPUAL_REPACK_INPUT_LAYOUT", "YES"} },
    { {"VPUX_VPUAL_REPACK_INPUT_LAYOUT", "NO"} },
    #endif
    { {"VPUX_PLATFORM", "AUTO"} },
    { {"VPUX_PLATFORM", "VPU3400_A0"} },
};

static const std::vector<ConfigMap> colorFormatConfigs = {
    { },
    { {"VPUX_GRAPH_COLOR_FORMAT", "RGB"} },
    { {"VPUX_GRAPH_COLOR_FORMAT", "BGR"} }
};

static const std::vector<ConfigMap> preProcConfigs = {
    { },
    { {"VPUX_USE_M2I", "YES"}, {"VPUX_USE_SIPP", "NO"}, {"VPUX_USE_SHAVE_ONLY_M2I", "NO"} },
    { {"VPUX_USE_M2I", "NO"}, {"VPUX_USE_SIPP", "YES"}, {"VPUX_USE_SHAVE_ONLY_M2I", "NO"} },
    { {"VPUX_USE_M2I", "NO"}, {"VPUX_USE_SIPP", "NO"}, {"VPUX_USE_SHAVE_ONLY_M2I", "YES"} },
};

static const std::vector<ConfigMap> preProcParamConfigs = {
    { },
    { {"VPUX_PREPROCESSING_SHAVES", "4"}, {"VPUX_PREPROCESSING_LPI", "8"} },
    { {"VPUX_PREPROCESSING_SHAVES", "2"}, {"VPUX_PREPROCESSING_LPI", "4"} }
};

static const std::vector<ConfigMap> inferShavesConfigs = {
    { },
    { {"VPUX_INFERENCE_SHAVES", "2"} },
    { {"VPUX_INFERENCE_SHAVES", "4"} },
    #if defined(__arm) || defined(__aarch64__)
    { {"VPUX_VPUAL_INFERENCE_SHAVES", "2"} },
    { {"VPUX_VPUAL_INFERENCE_SHAVES", "4"} },
    #endif
};

static const auto allBaseConfigurations = ::testing::Combine(
    ::testing::ValuesIn(compileConfigs),
    ::testing::ValuesIn(baseInferConfigs),
    ::testing::ValuesIn(emptyConfigs),
    ::testing::ValuesIn(emptyConfigs),
    ::testing::ValuesIn(emptyConfigs),
    ::testing::ValuesIn(emptyConfigs));

INSTANTIATE_TEST_CASE_P(SomeCase1, KmbConfigTest, allBaseConfigurations);

static const auto allPreProcConfigurations = ::testing::Combine(
    ::testing::ValuesIn(emptyConfigs),
    ::testing::ValuesIn(emptyConfigs),
    ::testing::ValuesIn(colorFormatConfigs),
    ::testing::ValuesIn(preProcConfigs),
    ::testing::ValuesIn(preProcParamConfigs),
    ::testing::ValuesIn(inferShavesConfigs));

INSTANTIATE_TEST_CASE_P(SomeCase2, KmbConfigTest, allPreProcConfigurations);
