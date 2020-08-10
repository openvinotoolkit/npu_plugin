// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"
#include "kmb_test_tool.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsUtils {

#ifndef __aarch64__
// might need to use CommonTestUtils::DEVICE_CPU for ref calc
const TargetDevice testPlatformTargetDevice(CommonTestUtils::DEVICE_KEEMBAY);
#else
const TargetDevice testPlatformTargetDevice(CommonTestUtils::DEVICE_KEEMBAY);
#endif

const KmbTestEnvConfig KmbLayerTestsCommon::envConfig;

KmbLayerTestsCommon::KmbLayerTestsCommon(): kmbTestTool(envConfig) {
    if (!envConfig.IE_KMB_TESTS_LOG_LEVEL.empty()) {
        configuration[CONFIG_KEY(LOG_LEVEL)] = envConfig.IE_KMB_TESTS_LOG_LEVEL;
    }
    // todo: values are temporarily overriden to enable mcm compilation
    // targetDevice = testPlatformTargetDevice;
    inLayout = InferenceEngine::Layout::NHWC;
    outLayout = InferenceEngine::Layout::NHWC;
    inPrc = InferenceEngine::Precision::U8;
    outPrc = InferenceEngine::Precision::FP16;
}

void KmbLayerTestsCommon::BuildNetworkWithoutCompile() {
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    ConfigureNetwork();
}

void KmbLayerTestsCommon::ImportNetwork() {
    executableNetwork = kmbTestTool.importNetwork(getCore(),
        filesysTestName(testing::UnitTest::GetInstance()->current_test_info()));
}

void KmbLayerTestsCommon::ExportNetwork() {
    kmbTestTool.exportNetwork(executableNetwork, filesysTestName(testing::UnitTest::GetInstance()->current_test_info()));
}

void KmbLayerTestsCommon::Validate() {
    std::cout << "LayerTestsCommon::Validate()" << std::endl;
    LayerTestsCommon::Validate();
}

std::vector<std::vector<std::uint8_t>> KmbLayerTestsCommon::CalculateRefs() {
    std::cout << "LayerTestsCommon::CalculateRefs() beg" << std::endl;
    auto res = LayerTestsCommon::CalculateRefs();
    std::cout << "LayerTestsCommon::CalculateRefs() end" << std::endl;
    return res;
}

void KmbLayerTestsCommon::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    ConfigurePlugin();
    std::cout << "KmbLayerTestsCommon::BuildNetworkWithoutCompile" << std::endl;
    BuildNetworkWithoutCompile();
#ifndef __aarch64__
    std::cout << "KmbLayerTestsCommon::Compile" << std::endl;
    executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice);
    std::cout << "KmbLayerTestsCommon::ExportNetwork()" << std::endl;
    ASSERT_NO_THROW(ExportNetwork());
#else
    std::cout << "KmbLayerTestsCommon::ImportNetwork()" << std::endl;
    ASSERT_NO_THROW(ImportNetwork());
    if (envConfig.IE_KMB_TESTS_RUN_INFER) {
        // todo: infers are not run forcefully; layer test networks hang the board
        SKIP() << "Skip infer due to layer test networks hang the board";
        std::cout << "KmbLayerTestsCommon::Infer()" << std::endl;
        Infer();
        std::cout << "KmbLayerTestsCommon::Validate()" << std::endl;
        Validate();
    } else {
        std::cout << "Skip KmbLayerTestsCommon::Infer()" << std::endl;
    }
#endif
}

}  // namespace LayerTestsUtils
