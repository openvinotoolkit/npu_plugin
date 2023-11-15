//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "common/functions.h"
#include "test_model/kmb_test_base.hpp"
#include "vpux_private_config.hpp"

using ConfigMap = std::map<std::string, std::string>;
using ConfigTestParams = std::tuple<ConfigMap, ConfigMap, ConfigMap, ConfigMap, ConfigMap, ConfigMap>;

class KmbConfigTest : public KmbLayerTestBase, public testing::WithParamInterface<ConfigTestParams> {
public:
    void runTest(const ConfigMap& compileConfig, const ConfigMap& inferConfig);
};

void KmbConfigTest::runTest(const ConfigMap& compileConfig, const ConfigMap& inferConfig) {
    try {
        TestNetwork testNet;
        testNet.setUserInput("input", Precision::FP16, Layout::NCHW)
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
    } catch (const import_error& ex) {
        std::cerr << ex.what() << std::endl;
        GTEST_SKIP() << ex.what();
    }
}

TEST_P(KmbConfigTest, setConfig) {
    // [Track number: #E20189]
    SKIP_ON("LEVEL0", "Sporadic failures on device");
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
    if (inferConfig.find(VPUX_CONFIG_KEY(COMPILER_TYPE)) == inferConfig.end())
        inferConfig[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);

    runTest(compileConfig, inferConfig);
}

static const std::vector<ConfigMap> emptyConfigs = {{}};

static const std::vector<ConfigMap> compileConfigs = {{}};

static const std::vector<ConfigMap> baseInferConfigs = {
        {},
        {{"PERF_COUNT", "YES"}},
        {{"PERF_COUNT", "NO"}},
        {{"DEVICE_ID", ""}},
        {{"NPU_PLATFORM", PlatformEnvironment::PLATFORM}},
};

static const auto allBaseConfigurations = ::testing::Combine(
        ::testing::ValuesIn(compileConfigs), ::testing::ValuesIn(baseInferConfigs), ::testing::ValuesIn(emptyConfigs),
        ::testing::ValuesIn(emptyConfigs), ::testing::ValuesIn(emptyConfigs), ::testing::ValuesIn(emptyConfigs));

INSTANTIATE_TEST_SUITE_P(DISABLED_TMP_SomeCase1, KmbConfigTest, allBaseConfigurations);
