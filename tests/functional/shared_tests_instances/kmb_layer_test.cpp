// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"

#include <ie_utils.hpp>
#include <transformations/op_conversions/convert_batch_to_space.hpp>
#include <transformations/op_conversions/convert_space_to_batch.hpp>

#include "kmb_test_tool.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

namespace LayerTestsUtils {

#ifndef __aarch64__
// might need to use CommonTestUtils::DEVICE_CPU for ref calc
const TargetDevice testPlatformTargetDevice("KMB");
#else
const TargetDevice testPlatformTargetDevice("KMB");
#endif

const KmbTestEnvConfig KmbLayerTestsCommon::envConfig;

KmbLayerTestsCommon::KmbLayerTestsCommon(): kmbTestTool(envConfig) {
    if (!envConfig.IE_KMB_TESTS_LOG_LEVEL.empty()) {
        configuration[CONFIG_KEY(LOG_LEVEL)] = envConfig.IE_KMB_TESTS_LOG_LEVEL;
    }
}

void KmbLayerTestsCommon::BuildNetworkWithoutCompile() {
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    ConfigureNetwork();
}

void KmbLayerTestsCommon::ImportNetwork() {
    executableNetwork = kmbTestTool.importNetwork(getCore(),
        filesysName(testing::UnitTest::GetInstance()->current_test_info()));
}

void KmbLayerTestsCommon::ExportNetwork() {
    kmbTestTool.exportNetwork(executableNetwork, filesysName(testing::UnitTest::GetInstance()->current_test_info()));
}

void KmbLayerTestsCommon::Validate() {
    std::cout << "LayerTestsCommon::Validate()" << std::endl;
    LayerTestsCommon::Validate();
}

void KmbLayerTestsCommon::Compare(const std::vector<std::vector<std::uint8_t>>& expectedOutputs, const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs) {
    for (std::size_t outputIndex = 0; outputIndex < expectedOutputs.size(); ++outputIndex) {
        const auto& expected = expectedOutputs[outputIndex];
        const auto& actual = actualOutputs[outputIndex];

        // TODO: The compare function only supports I32 and FP32 precision.
        switch (actual->getTensorDesc().getPrecision()) {
            case InferenceEngine::Precision::FP16:
                LayerTestsCommon::Compare(expected, toPrecision(actual, InferenceEngine::Precision::FP32));
                break;
            case InferenceEngine::Precision::U8:
                LayerTestsCommon::Compare(expected, toPrecision(actual, InferenceEngine::Precision::I32));
                break;
            default:
                LayerTestsCommon::Compare(expected, actual);
        }
    }
}

std::vector<std::vector<std::uint8_t>> KmbLayerTestsCommon::CalculateRefs() {
    std::cout << "LayerTestsCommon::CalculateRefs() beg" << std::endl;

    // TODO: The calculate reference function not support FP16 precision.
    //       The compare function only supports I32 and FP32 precision.
    for (auto& input_blob : inputs) {
        if (input_blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::FP16) {
            input_blob = toPrecision(input_blob, InferenceEngine::Precision::FP32);
        }
        if (input_blob->getTensorDesc().getPrecision() == InferenceEngine::Precision::U8) {
            input_blob = toPrecision(input_blob, InferenceEngine::Precision::I32);
        }
    }

    if (outPrc == InferenceEngine::Precision::UNSPECIFIED) {
        auto actualOutputs = GetOutputs();
        outPrc = actualOutputs[0]->getTensorDesc().getPrecision();
    }
    if (outPrc == InferenceEngine::Precision::FP16) {
        outPrc = InferenceEngine::Precision::FP32;
    }
    if (outPrc == InferenceEngine::Precision::U8) {
        outPrc = InferenceEngine::Precision::I32;
    }

    auto res = LayerTestsCommon::CalculateRefs();
    std::cout << "LayerTestsCommon::CalculateRefs() end" << std::endl;

    return res;
}

void KmbLayerTestsCommon::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::cout << "KmbLayerTestsCommon::BuildNetworkWithoutCompile" << std::endl;
    BuildNetworkWithoutCompile();
#ifndef __aarch64__
    std::cout << "KmbLayerTestsCommon::Compile" << std::endl;
    executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration);
    std::cout << "KmbLayerTestsCommon::ExportNetwork()" << std::endl;
    ASSERT_NO_THROW(ExportNetwork());
#else
    try {
        std::cout << "KmbLayerTestsCommon::ImportNetwork()" << std::endl;
        SkipBeforeImport();
        ImportNetwork();
        if (envConfig.IE_KMB_TESTS_RUN_INFER) {
            std::cout << "KmbLayerTestsCommon::Infer()" << std::endl;
            SkipBeforeInfer();
            Infer();
            std::cout << "KmbLayerTestsCommon::Validate()" << std::endl;
            SkipBeforeValidate();
            Validate();
        } else {
            std::cout << "Skip KmbLayerTestsCommon::Infer()" << std::endl;
        }
    } catch (const KmbSkipTestException &e) {
        std::cout << "Skipping the test due to: " << e.what() << std::endl;
        SKIP() << "Skipping the test due to: " << e.what();
    }
#endif
}

}  // namespace LayerTestsUtils
