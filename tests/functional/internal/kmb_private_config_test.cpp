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

using ConfigTestParams = std::tuple<std::map<std::string, std::string>, std::map<std::string, std::string>>;

class KmbPrivateConfigTest : public KmbNetworkTestBase, public testing::WithParamInterface<ConfigTestParams> {
public:
    void runTest(
        const std::map<std::string, std::string>& compileConfig,
        const std::map<std::string, std::string>& inferConfig);
};

void KmbPrivateConfigTest::runTest(
        const std::map<std::string, std::string>& compileConfig,
        const std::map<std::string, std::string>& inferConfig) {
    if (RUN_COMPILER) {
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
        CNNNetwork cnnNet = testNet.getCNNNetwork();
        ExecutableNetwork exeNet = core->LoadNetwork(cnnNet, DEVICE_NAME, compileConfig);
        KmbTestBase::exportNetwork(exeNet);
    }

    if (RUN_INFER) {
        ExecutableNetwork importedNet = KmbTestBase::importNetwork(inferConfig);
        constexpr bool printTime = false;
        const BlobMap inputs = {};
        KmbTestBase::runInfer(importedNet, inputs, printTime);
    }
}

TEST_P(KmbPrivateConfigTest, setConfig) {
    const auto& param = GetParam();
    const auto compileConfig = std::get<0>(param);
    const auto inferConfig = std::get<1>(param);
    runTest(compileConfig, inferConfig);
}

static const std::vector<std::map<std::string, std::string>> compileConfigs = {
    {}
};

static const std::vector<std::map<std::string, std::string>> inferConfigs = {
    { {"VPU_KMB_USE_CORE_NN", "YES"} },
    { {"VPU_KMB_USE_CORE_NN", "NO"} },
};

static auto allConfigurations = ::testing::Combine(
    ::testing::ValuesIn(compileConfigs),
    ::testing::ValuesIn(inferConfigs));

INSTANTIATE_TEST_CASE_P(SomeCase, KmbPrivateConfigTest, allConfigurations);
