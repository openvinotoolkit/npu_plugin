// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/softmax.hpp"

#include <vector>

#include "kmb_layer_test.hpp"

namespace BehaviorTestsDefinitions {
using namespace LayerTestsDefinitions;

class KmbSwapDeviceNOutbutBlobLayerTest: public SoftMaxLayerTest, virtual public LayerTestsUtils::KmbLayerTestsCommon {
public:
    void Run() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        BuildNetworkWithoutCompile();

        if (envConfig.IE_KMB_TESTS_RUN_COMPILER) {
            ASSERT_NO_THROW(executableNetwork = getCore()->LoadNetwork(cnnNetwork, LayerTestsUtils::testPlatformTargetDevice, configuration));
            if (envConfig.IE_KMB_TESTS_RUN_EXPORT) {
                ASSERT_NO_THROW(ExportNetwork());
            }
        } else {
            ImportNetwork();
        }
        GenerateInputs();

        if (envConfig.IE_KMB_TESTS_RUN_INFER) {
            Infer();
            Validate();
        }
    }
    void Infer() override {
        inferRequest = executableNetwork.CreateInferRequest();

        const auto& inputsInfo = executableNetwork.GetInputsInfo();
        const auto& functionParams = function->get_parameters();
        for (int i = 0; i < functionParams.size(); ++i) {
            const auto& param = functionParams[i];
            const auto infoIt = inputsInfo.find(param->get_friendly_name());
            GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

            const auto& info = infoIt->second;
            auto blob = inputs[i];
            inferRequest.SetBlob(info->name(), blob);
        }
        if (configuration.count(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED) &&
            configuration.count(InferenceEngine::PluginConfigParams::YES)) {
            auto batchSize = executableNetwork.GetInputsInfo().begin()->second->getTensorDesc().getDims()[0] / 2;
            inferRequest.SetBatch(batchSize);
        }

        std::map<std::string, InferenceEngine::Blob::Ptr> outputBlobs;

        for (auto outInfo: executableNetwork.GetOutputsInfo()) {
            outputBlobs[outInfo.first] = inferRequest.GetBlob(outInfo.first);
        }

        inferRequest.Infer();

        for (auto outInfo: executableNetwork.GetOutputsInfo()) {
            // Verify that the output blob has changed the memory area it is pointing to.
            // This pointer swap is implementing optimization and is subject to change.
            // Having this assertion failed means that aforementioned optimization has not been applied.
            ASSERT_NE(outputBlobs[outInfo.first].get(), inferRequest.GetBlob(outInfo.first).get());
        }
    }
};

TEST_P(KmbSwapDeviceNOutbutBlobLayerTest, InferAndCompareWithRefs) {
    useCompilerMLIR();
    Run();
}

}  // namespace BehaviorTestsDefinitions

using namespace ngraph::helpers;
using namespace BehaviorTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Layout> inLayouts2D = {
    InferenceEngine::Layout::NC,
};

const std::vector<InferenceEngine::SizeVector> inShapes2D = {
    InferenceEngine::SizeVector {1, 100},
};

const std::vector<size_t> axis2D = {
    0, 1
};

const auto params2D = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::ValuesIn(inLayouts2D),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inShapes2D),
    testing::ValuesIn(axis2D),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
    smoke_BehaviorTest_SoftMax2D,
    KmbSwapDeviceNOutbutBlobLayerTest,
    params2D,
    SoftMaxLayerTest::getTestCaseName
);

const std::vector<InferenceEngine::SizeVector> inShapes4D = {
    InferenceEngine::SizeVector {1, 100, 1, 1},
};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4D = testing::Combine(
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::NCHW),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::ValuesIn(inShapes4D),
    testing::ValuesIn(axis4D),
    testing::Values(LayerTestsUtils::testPlatformTargetDevice),
    testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_CASE_P(
    smoke_BehaviorTest_SoftMax4D,
    KmbSwapDeviceNOutbutBlobLayerTest,
    params4D,
    SoftMaxLayerTest::getTestCaseName
);

}  // namespace
