// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"
#include "kmb_test_report.hpp"
#include "vpux_private_config.hpp"

#include "kmb_test_tool.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "vpux/utils/core/format.hpp"

namespace LayerTestsUtils {

const TargetDevice testPlatformTargetDevice("VPUX");
const auto DEFAULT_IE_KMB_TESTS_INFERENCE_SHAVES = "16";

const KmbTestEnvConfig KmbLayerTestsCommon::envConfig;

KmbLayerTestsCommon::KmbLayerTestsCommon(): kmbTestTool(envConfig) {
    IE_ASSERT(core != nullptr);

    if (!envConfig.IE_KMB_TESTS_LOG_LEVEL.empty()) {
        core->SetConfig({{CONFIG_KEY(LOG_LEVEL), envConfig.IE_KMB_TESTS_LOG_LEVEL}}, testPlatformTargetDevice);
    }
}

void KmbLayerTestsCommon::BuildNetworkWithoutCompile() {
    cnnNetwork = InferenceEngine::CNNNetwork{function};

    configuration[VPUX_CONFIG_KEY(INFERENCE_SHAVES)] = DEFAULT_IE_KMB_TESTS_INFERENCE_SHAVES;
    configuration[VPUX_CONFIG_KEY(PLATFORM)] = envConfig.IE_KMB_TESTS_PLATFORM;
    ConfigureNetwork();
}

void KmbLayerTestsCommon::ImportNetwork() {
    IE_ASSERT(core != nullptr);
    executableNetwork = kmbTestTool.importNetwork(core,
        filesysName(testing::UnitTest::GetInstance()->current_test_info(), ".net", !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
}

void KmbLayerTestsCommon::ExportNetwork() {
    kmbTestTool.exportNetwork(executableNetwork,
        filesysName(testing::UnitTest::GetInstance()->current_test_info(), ".net", !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
}

void KmbLayerTestsCommon::ExportInput() {
    int i = 0;
    for (const auto &input : executableNetwork.GetInputsInfo()) {
        const auto &info = input.second;
        const auto ext = llvm::formatv(".{0}.{1}", info->name(), "in").str();
        kmbTestTool.exportBlob(inputs[i++],
            filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext, !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
    }
}

void KmbLayerTestsCommon::ExportOutput() {
    int i = 0;
    const auto & outputs = GetOutputs();
    for (const auto &output : executableNetwork.GetOutputsInfo()) {
        const auto &info = output.second;
        const auto ext = llvm::formatv(".{0}.{1}", info->getName(), "out").str();
        kmbTestTool.exportBlob(outputs[i++],
            filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext, !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
    }
}

void KmbLayerTestsCommon::ImportInput() {
    // infer request should be adapted afterwards
    int i = 0;
    for (const auto &input : executableNetwork.GetInputsInfo()) {
        const auto &info = input.second;
        const auto ext = llvm::formatv(".{0}.{1}", info->name(), "in").str();
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info->getTensorDesc());
        blob->allocate();
        kmbTestTool.importBlob(blob,
            filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext, !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
        inputs[i++] = blob;
    }
}

void KmbLayerTestsCommon::ExportReference(
        const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& refs) {
    size_t i = 0;
    for (const auto &output : executableNetwork.GetOutputsInfo()) {
        const auto &name = output.first;

        auto& ref = refs[i++];
        auto referenceBlob = InferenceEngine::make_shared_blob<uint8_t>(
            InferenceEngine::TensorDesc{
                InferenceEngine::Precision::U8,
                InferenceEngine::SizeVector{ref.second.size()},
                InferenceEngine::Layout::C
            }, const_cast<std::uint8_t*>(&ref.second[0]), ref.second.size());
        const auto ext = llvm::formatv(".{0}.{1}", name, "ref").str();
        kmbTestTool.exportBlob(referenceBlob,
            filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext, !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
    }
}

void KmbLayerTestsCommon::ImportReference(
        const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& refs) {
    size_t i = 0;
    for (const auto &output : executableNetwork.GetOutputsInfo()) {
        const auto &name = output.first;

        auto& ref = refs[i++];
        auto referenceBlob = InferenceEngine::make_shared_blob<uint8_t>(
            InferenceEngine::TensorDesc{
                InferenceEngine::Precision::U8,
                InferenceEngine::SizeVector{ref.second.size()},
                InferenceEngine::Layout::C
            }, const_cast<uint8_t *>(ref.second.data()), ref.second.size());
        const auto ext = llvm::formatv(".{0}.{1}", name, "ref").str();
        kmbTestTool.importBlob(referenceBlob,
            filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext, !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
    }
}

void KmbLayerTestsCommon::Validate() {
    std::cout << "LayerTestsCommon::Validate()" << std::endl;

    const auto rearrangeDataToLayoutFromNumDims = [](std::vector<InferenceEngine::Blob::Ptr>& blobs) {
        for (int i = 0; i < blobs.size(); ++i) {
            const auto defaultLayout = InferenceEngine::TensorDesc::getLayoutByDims(blobs[i]->getTensorDesc().getDims());
            if (blobs[i]->getTensorDesc().getLayout() != defaultLayout) {
                blobs[i] = FuncTestUtils::convertBlobLayout(blobs[i], defaultLayout);
            }
        }
    };

    // TODO:#-52272 Move re-layout to the CalculateRefs method
    rearrangeDataToLayoutFromNumDims(inputs);

    auto expectedOutputs = CalculateRefs();
    auto actualOutputs = GetOutputs();

    if (envConfig.IE_KMB_TESTS_IMPORT_REF) {
        std::cout << "KmbLayerTestsCommon::ImportReference()" << std::endl;
        ImportReference(expectedOutputs);
    }

    if (expectedOutputs.empty()) {
        return;
    }

    IE_ASSERT(actualOutputs.size() == expectedOutputs.size())
        << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

    rearrangeDataToLayoutFromNumDims(actualOutputs);
    Compare(expectedOutputs, actualOutputs);
}

void KmbLayerTestsCommon::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    std::cout << "KmbLayerTestsCommon::BuildNetworkWithoutCompile" << std::endl;
    BuildNetworkWithoutCompile();
    KmbTestReport& report = KmbTestReport::getInstance();
    const auto& testInfo = testing::UnitTest::GetInstance()->current_test_info();
    report.run(testInfo);

    try {
        if (envConfig.IE_KMB_TESTS_RUN_COMPILER) {
            std::cout << "KmbLayerTestsCommon::Compile" << std::endl;
            SkipBeforeLoad();

            ASSERT_NO_THROW(executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration));
            report.compiled(testInfo);

            if (envConfig.IE_KMB_TESTS_RUN_EXPORT) {
                std::cout << "KmbLayerTestsCommon::ExportNetwork()" << std::endl;
                ASSERT_NO_THROW(ExportNetwork());
            }
        } else {
            std::cout << "KmbLayerTestsCommon::ImportNetwork()" << std::endl;
            SkipBeforeLoad();
            SkipBeforeImport();
            ImportNetwork();
            report.imported(testInfo);
        }
        GenerateInputs();
        if (envConfig.IE_KMB_TESTS_EXPORT_INPUT) {
            std::cout << "KmbLayerTestsCommon::ExportInput()" << std::endl;
            ExportInput();
        }
        if (envConfig.IE_KMB_TESTS_IMPORT_INPUT) {
            std::cout << "KmbLayerTestsCommon::ImportInput()" << std::endl;
            ImportInput();
        }
        if (envConfig.IE_KMB_TESTS_RUN_INFER) {
            std::cout << "KmbLayerTestsCommon::Infer()" << std::endl;
            SkipBeforeInfer();
            Infer();
            report.inferred(testInfo);
        }
        if (envConfig.IE_KMB_TESTS_EXPORT_REF) {
            std::cout << "KmbLayerTestsCommon::ExportReference()" << std::endl;
            ExportReference(CalculateRefs());
        }
        if (envConfig.IE_KMB_TESTS_EXPORT_OUTPUT) {
            std::cout << "KmbLayerTestsCommon::ExportOutput()" << std::endl;
            ExportOutput();
        }
        if (envConfig.IE_KMB_TESTS_RUN_INFER) {
            std::cout << "KmbLayerTestsCommon::Validate()" << std::endl;
            SkipBeforeValidate();
            Validate();
            report.validated(testInfo);
        } else {
            std::cout << "Skip KmbLayerTestsCommon::Infer()" << std::endl;
        }
    } catch (const KmbSkipTestException &e) {
        std::cout << "Skipping the test due to: " << e.what() << std::endl;
        report.skipped(testInfo);
        GTEST_SKIP() << "Skipping the test due to: " << e.what();
    }
}

void KmbLayerTestsCommon::useCompilerMLIR() {
    configuration[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);
}

void KmbLayerTestsCommon::setReferenceHardwareModeMLIR() {
    configuration[VPUX_CONFIG_KEY(COMPILATION_MODE)] = "ReferenceHW";
}

bool KmbLayerTestsCommon::isCompilerMCM() const {
    const auto it = configuration.find(VPUX_CONFIG_KEY(COMPILER_TYPE));
    if (it == configuration.end()) {
        // Default value for COMPILER_TYPE is MCM
        return true;
    }

    return it->second == VPUX_CONFIG_VALUE(MCM);
}

bool KmbLayerTestsCommon::isCompilerMLIR() const {
    const auto it = configuration.find(VPUX_CONFIG_KEY(COMPILER_TYPE));
    if (it == configuration.end()) {
        // Default value for COMPILER_TYPE is MCM
        return false;
    }

    return it->second == VPUX_CONFIG_VALUE(MLIR);
}

void KmbLayerTestsCommon::disableMcmPasses(const std::vector<std::pair<std::string, std::string>>& banList) {
    const auto passFold = [](std::string list, const std::pair<std::string, std::string>& pass) {
        return std::move(list) + pass.first + "," + pass.second + ";";
    };

    auto configValue = std::accumulate(begin(banList), end(banList), std::string{}, passFold);
    configValue.pop_back();

    configuration[VPU_COMPILER_CONFIG_KEY(COMPILATION_PASS_BAN_LIST)] = std::move(configValue);
}

void KmbLayerTestsCommon::TearDown() {
    LayerTestsCommon::TearDown();
    PluginCache::get().reset();
}
}  // namespace LayerTestsUtils
