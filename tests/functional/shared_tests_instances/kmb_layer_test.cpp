// Copyright (C) Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kmb_layer_test.hpp"
#include "kmb_test_report.hpp"
#include "vpux_private_config.hpp"

#include "functional_test_utils/blob_utils.hpp"
#include "kmb_test_tool.hpp"

#include <common/functions.h>
#include "vpux/utils/core/format.hpp"

namespace LayerTestsUtils {

const KmbTestEnvConfig& KmbLayerTestsCommon::envConfig = KmbTestEnvConfig::getInstance();

const TargetDevice testPlatformTargetDevice = KmbTestEnvConfig::getInstance().IE_KMB_TESTS_DEVICE_NAME.empty()
                                                      ? CommonTestUtils::DEVICE_KEEMBAY
                                                      : KmbTestEnvConfig::getInstance().IE_KMB_TESTS_DEVICE_NAME;

KmbLayerTestsCommon::KmbLayerTestsCommon(): kmbTestTool(envConfig) {
    IE_ASSERT(core != nullptr);

    const std::string configDevice = testPlatformTargetDevice.substr(0, testPlatformTargetDevice.find("."));

    if (!envConfig.IE_KMB_TESTS_LOG_LEVEL.empty()) {
        core->SetConfig({{CONFIG_KEY(LOG_LEVEL), envConfig.IE_KMB_TESTS_LOG_LEVEL}}, configDevice);
    }
}

void KmbLayerTestsCommon::BuildNetworkWithoutCompile() {
    cnnNetwork = InferenceEngine::CNNNetwork{function};

    if (configuration.count(VPUX_CONFIG_KEY(PLATFORM)) == 0) {
        configuration[VPUX_CONFIG_KEY(PLATFORM)] = envConfig.IE_KMB_TESTS_PLATFORM;
    }
    if (configuration.find(VPUX_CONFIG_KEY(COMPILER_TYPE)) == configuration.end())
        configuration[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);
    ConfigureNetwork();
}

void KmbLayerTestsCommon::ImportNetwork() {
    IE_ASSERT(core != nullptr);
    executableNetwork =
            kmbTestTool.importNetwork(core, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ".net",
                                                        !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
}

void KmbLayerTestsCommon::ExportNetwork() {
    kmbTestTool.exportNetwork(executableNetwork, filesysName(testing::UnitTest::GetInstance()->current_test_info(),
                                                             ".net", !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
}

void KmbLayerTestsCommon::ExportInput() {
    int i = 0;
    for (const auto& input : executableNetwork.GetInputsInfo()) {
        const auto& info = input.second;
        const auto ext = vpux::printToString(".{0}.{1}", info->name(), "in");
        kmbTestTool.exportBlob(inputs[i++], filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                                        !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
    }
}

void KmbLayerTestsCommon::ExportOutput() {
    int i = 0;
    const auto& outputs = GetOutputs();
    for (const auto& output : executableNetwork.GetOutputsInfo()) {
        const auto& info = output.second;
        const auto ext = vpux::printToString(".{0}.{1}", info->getName(), "out");
        kmbTestTool.exportBlob(outputs[i++], filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                                         !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
    }
}

void KmbLayerTestsCommon::ImportInput() {
    // infer request should be adapted afterwards
    int i = 0;
    for (const auto& input : executableNetwork.GetInputsInfo()) {
        const auto& info = input.second;
        const auto ext = vpux::printToString(".{0}.{1}", info->name(), "in");
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info->getTensorDesc());
        blob->allocate();
        kmbTestTool.importBlob(blob, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                                 !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
        inputs[i++] = blob;
    }
}

void KmbLayerTestsCommon::ExportReference(
        const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& refs) {
    size_t i = 0;
    for (const auto& output : executableNetwork.GetOutputsInfo()) {
        const auto& name = output.first;

        auto& ref = refs[i++];
        auto referenceBlob = InferenceEngine::make_shared_blob<uint8_t>(
                InferenceEngine::TensorDesc{InferenceEngine::Precision::U8,
                                            InferenceEngine::SizeVector{ref.second.size()}, InferenceEngine::Layout::C},
                const_cast<std::uint8_t*>(&ref.second[0]), ref.second.size());
        const auto ext = vpux::printToString(".{0}.{1}", name, "ref");
        kmbTestTool.exportBlob(referenceBlob, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                                          !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
    }
}

void KmbLayerTestsCommon::ImportReference(
        const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& refs) {
    size_t i = 0;
    for (const auto& output : executableNetwork.GetOutputsInfo()) {
        const auto& name = output.first;

        auto& ref = refs[i++];
        auto referenceBlob = InferenceEngine::make_shared_blob<uint8_t>(
                InferenceEngine::TensorDesc{InferenceEngine::Precision::U8,
                                            InferenceEngine::SizeVector{ref.second.size()}, InferenceEngine::Layout::C},
                const_cast<uint8_t*>(ref.second.data()), ref.second.size());
        const auto ext = vpux::printToString(".{0}.{1}", name, "ref");
        kmbTestTool.importBlob(referenceBlob, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                                          !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
    }
}

std::vector<InferenceEngine::Blob::Ptr> KmbLayerTestsCommon::ImportOutputs() {
    std::vector<InferenceEngine::Blob::Ptr> outputs = std::vector<InferenceEngine::Blob::Ptr>{};
    InferenceEngine::ConstOutputsDataMap mapInfo = executableNetwork.GetOutputsInfo();
    std::vector<std::pair<std::string, InferenceEngine::CDataPtr>> outputsInfo(mapInfo.begin(), mapInfo.end());

    for (const auto& output : outputsInfo) {
        const auto ext = vpux::printToString(".{0}.{1}", output.first, "out");

        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(output.second->getTensorDesc());
        blob->allocate();

        kmbTestTool.importBlob(blob, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                                 !envConfig.IE_KMB_TESTS_LONG_FILE_NAME));
        outputs.push_back(blob);
    }

    return outputs;
}

void KmbLayerTestsCommon::Validate() {
    std::cout << "LayerTestsCommon::Validate()" << std::endl;

    const auto rearrangeDataToLayoutFromNumDims = [](std::vector<InferenceEngine::Blob::Ptr>& blobs) {
        for (int i = 0; i < blobs.size(); ++i) {
            const auto defaultLayout =
                    InferenceEngine::TensorDesc::getLayoutByDims(blobs[i]->getTensorDesc().getDims());
            if (blobs[i]->getTensorDesc().getLayout() != defaultLayout) {
                blobs[i] = FuncTestUtils::convertBlobLayout(blobs[i], defaultLayout);
            }
        }
    };

    const auto convertDataToFP32 = [](std::vector<InferenceEngine::Blob::Ptr>& blobs) {
        for (int i = 0; i < blobs.size(); ++i) {
            if (blobs[i]->getTensorDesc().getPrecision() != InferenceEngine::Precision::FP32) {
                blobs[i] = FuncTestUtils::copyBlobWithCast<InferenceEngine::Precision::FP32>(blobs[i]);
            }
        }
    };

    // TODO:#-52272 Move re-layout to the CalculateRefs method
    rearrangeDataToLayoutFromNumDims(inputs);

    auto expectedOutputs = CalculateRefs();

    std::vector<InferenceEngine::Blob::Ptr> actualOutputs;

    if (envConfig.IE_KMB_TESTS_RUN_INFER) {
        actualOutputs = GetOutputs();
    }

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

    // TODO: Remove after C#-101214
    convertDataToFP32(actualOutputs);

    Compare(expectedOutputs, actualOutputs);
}

void KmbLayerTestsCommon::Run() {
    ov::test::utils::PassRate::Statuses status = FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()
                                                         ? ov::test::utils::PassRate::Statuses::SKIPPED
                                                         : ov::test::utils::PassRate::Statuses::CRASHED;
    auto& summary = ov::test::utils::OpSummary::getInstance();
    summary.setDeviceName(targetDevice);
    summary.updateOPsStats(function, status);

    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    functionRefs = ngraph::clone_function(*function);

    std::cout << "KmbLayerTestsCommon::BuildNetworkWithoutCompile" << std::endl;
    BuildNetworkWithoutCompile();
    KmbTestReport& report = KmbTestReport::getInstance();
    const auto& testInfo = testing::UnitTest::GetInstance()->current_test_info();
    report.run(testInfo);

    std::string errorMessage;
    try {
        if (envConfig.IE_KMB_TESTS_RUN_COMPILER) {
            std::cout << "KmbLayerTestsCommon::Compile" << std::endl;
            std::ostringstream ostr;
            ostr << "LoadNetwork Config: ";
            for (const auto& item : configuration) {
                ostr << item.first << "=" << item.second << "; ";
            }
            std::cout << ostr.str() << std::endl;
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

        bool runInfer = envConfig.IE_KMB_TESTS_RUN_INFER;
        std::string runInferSkipReason = runInfer ? "-" : "environment variable value";

        const std::string backendName = getBackendName(*core);
        // turn off running infers forcefully for the cases:
        const auto noDevice = backendName.empty();
        if (runInfer && noDevice) {
            runInfer = false;
            runInferSkipReason = "backend is empty (no device)";
        }

        if (runInfer) {
            std::cout << "KmbLayerTestsCommon::Infer() with backend '" << backendName << "'" << std::endl;
            SkipBeforeInfer();
            Infer();
            report.inferred(testInfo);
        } else {
            std::cout << "Skip KmbLayerTestsCommon::Infer() due to: " << runInferSkipReason << std::endl;
        }
        if (envConfig.IE_KMB_TESTS_EXPORT_REF) {
            std::cout << "KmbLayerTestsCommon::ExportReference()" << std::endl;
            ExportReference(CalculateRefs());
        }
        if (envConfig.IE_KMB_TESTS_EXPORT_OUTPUT) {
            std::cout << "KmbLayerTestsCommon::ExportOutput()" << std::endl;
            ExportOutput();
        }
        if (runInfer) {
            std::cout << "KmbLayerTestsCommon::Validate()" << std::endl;
            SkipBeforeValidate();
            Validate();
            report.validated(testInfo);
        } else {
            std::cout << "Skip KmbLayerTestsCommon::Validate() due to: " << runInferSkipReason << std::endl;
        }
        status = ov::test::utils::PassRate::Statuses::PASSED;
    } catch (const KmbSkipTestException& e) {
        std::cout << "Skipping the test due to: " << e.what() << std::endl;
        report.skipped(testInfo);
        GTEST_SKIP() << "Skipping the test due to: " << e.what();
    } catch (const std::exception& ex) {
        status = ov::test::utils::PassRate::Statuses::FAILED;
        errorMessage = ex.what();
    } catch (...) {
        status = ov::test::utils::PassRate::Statuses::FAILED;
        errorMessage = "Unknown failure occurred.";
    }

    summary.updateOPsStats(function, status);
    if (status != ov::test::utils::PassRate::Statuses::PASSED) {
        std::cout << "Test has failed: " << errorMessage.c_str() << std::endl;
        GTEST_FATAL_FAILURE_(errorMessage.c_str());
    }
}

void KmbLayerTestsCommon::setReferenceSoftwareModeMLIR() {
    configuration[VPUX_CONFIG_KEY(COMPILATION_MODE)] = "ReferenceSW";
}

void KmbLayerTestsCommon::setDefaultHardwareModeMLIR() {
    configuration[VPUX_CONFIG_KEY(COMPILATION_MODE)] = "DefaultHW";
}

void KmbLayerTestsCommon::setPlatformVPU3700() {
    configuration[VPUX_CONFIG_KEY(PLATFORM)] = "VPU3700";
}

void KmbLayerTestsCommon::setPlatformVPU3720() {
    configuration[VPUX_CONFIG_KEY(PLATFORM)] = "VPU3720";
}

void KmbLayerTestsCommon::setSingleClusterMode() {
    configuration[VPUX_CONFIG_KEY(DPU_GROUPS)] = std::to_string(1);
    configuration[VPUX_CONFIG_KEY(DMA_ENGINES)] = std::to_string(1);
}

void KmbLayerTestsCommon::setPerformanceHintLatency() {
    configuration[CONFIG_KEY(PERFORMANCE_HINT)] = "LATENCY";
}

void KmbLayerTestsCommon::useELFCompilerBackend() {
    configuration[VPUX_CONFIG_KEY(USE_ELF_COMPILER_BACKEND)] = CONFIG_VALUE(YES);
}

void KmbLayerTestsCommon::TearDown() {
    LayerTestsCommon::TearDown();
    PluginCache::get().reset();
}
}  // namespace LayerTestsUtils
