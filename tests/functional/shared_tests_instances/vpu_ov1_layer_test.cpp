// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "vpu_ov1_layer_test.hpp"
#include "vpu_test_report.hpp"
#include "vpux_private_properties.hpp"

#include "common/utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include <common/functions.h>
#include "vpux/utils/core/format.hpp"

namespace LayerTestsUtils {

const VpuTestEnvConfig& VpuOv1LayerTestsCommon::envConfig = VpuTestEnvConfig::getInstance();

TargetDevice testPlatformTargetDevice() {
    static TargetDevice res = VpuTestEnvConfig::getInstance().IE_NPU_TESTS_DEVICE_NAME.empty()
                                      ? ov::test::utils::DEVICE_NPU
                                      : VpuTestEnvConfig::getInstance().IE_NPU_TESTS_DEVICE_NAME;
    return res;
}

VpuOv1LayerTestsCommon::VpuOv1LayerTestsCommon(): testTool(envConfig) {
    IE_ASSERT(core != nullptr);

    const std::string configDevice = testPlatformTargetDevice().substr(0, testPlatformTargetDevice().find("."));

    if (!envConfig.IE_NPU_TESTS_LOG_LEVEL.empty()) {
        core->SetConfig({{CONFIG_KEY(LOG_LEVEL), envConfig.IE_NPU_TESTS_LOG_LEVEL}}, configDevice);
    }
}

void VpuOv1LayerTestsCommon::BuildNetworkWithoutCompile() {
    cnnNetwork = InferenceEngine::CNNNetwork{function};

    if (configuration.count(ov::intel_vpux::platform.name()) == 0) {
        configuration[ov::intel_vpux::platform.name()] = envConfig.IE_NPU_TESTS_PLATFORM;
    }
    if (configuration.find(ov::intel_vpux::compiler_type.name()) == configuration.end())
        configuration[ov::intel_vpux::compiler_type.name()] = "MLIR";
    ;
    ConfigureNetwork();
}

void VpuOv1LayerTestsCommon::ImportNetwork() {
    IE_ASSERT(core != nullptr);
    executableNetwork = testTool.importNetwork(core, filesysName(testing::UnitTest::GetInstance()->current_test_info(),
                                                                 ".net", !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
}

void VpuOv1LayerTestsCommon::ExportNetwork() {
    testTool.exportNetwork(executableNetwork, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ".net",
                                                          !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
}

void VpuOv1LayerTestsCommon::ExportInput() {
    int i = 0;
    for (const auto& input : executableNetwork.GetInputsInfo()) {
        const auto& info = input.second;
        const auto ext = vpux::printToString(".{0}.{1}", info->name(), "in");
        testTool.exportBlob(inputs[i++], filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                                     !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
    }
}

void VpuOv1LayerTestsCommon::ExportOutput() {
    int i = 0;
    const auto& outputs = GetOutputs();
    for (const auto& output : executableNetwork.GetOutputsInfo()) {
        const auto& info = output.second;
        const auto ext = vpux::printToString(".{0}.{1}", info->getName(), "out");
        testTool.exportBlob(outputs[i++], filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                                      !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
    }
}

void VpuOv1LayerTestsCommon::ImportInput() {
    // infer request should be adapted afterwards
    int i = 0;
    for (const auto& input : executableNetwork.GetInputsInfo()) {
        const auto& info = input.second;
        const auto ext = vpux::printToString(".{0}.{1}", info->name(), "in");
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info->getTensorDesc());
        blob->allocate();
        testTool.importBlob(blob, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                              !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
        inputs[i++] = blob;
    }
}

void VpuOv1LayerTestsCommon::ExportReference(
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
        testTool.exportBlob(referenceBlob, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                                       !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
    }
}

void VpuOv1LayerTestsCommon::ImportReference(
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
        testTool.importBlob(referenceBlob, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                                       !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
    }
}

std::vector<InferenceEngine::Blob::Ptr> VpuOv1LayerTestsCommon::ImportOutputs() {
    std::vector<InferenceEngine::Blob::Ptr> outputs = std::vector<InferenceEngine::Blob::Ptr>{};
    InferenceEngine::ConstOutputsDataMap mapInfo = executableNetwork.GetOutputsInfo();
    std::vector<std::pair<std::string, InferenceEngine::CDataPtr>> outputsInfo(mapInfo.begin(), mapInfo.end());

    for (const auto& output : outputsInfo) {
        const auto ext = vpux::printToString(".{0}.{1}", output.first, "out");

        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(output.second->getTensorDesc());
        blob->allocate();

        testTool.importBlob(blob, filesysName(testing::UnitTest::GetInstance()->current_test_info(), ext,
                                              !envConfig.IE_NPU_TESTS_LONG_FILE_NAME));
        outputs.push_back(blob);
    }

    return outputs;
}

void VpuOv1LayerTestsCommon::Validate() {
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

    if (envConfig.IE_NPU_TESTS_RUN_INFER) {
        actualOutputs = GetOutputs();
    }

    if (envConfig.IE_NPU_TESTS_IMPORT_REF) {
        std::cout << "VpuOv1LayerTestsCommon::ImportReference()" << std::endl;
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

void VpuOv1LayerTestsCommon::Compare(
        const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
        const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs) {
    // Passing abs_threshold explicitly
    LayerTestsCommon::Compare(expectedOutputs, actualOutputs, threshold, abs_threshold);
}

void VpuOv1LayerTestsCommon::Run() {
    ov::test::utils::PassRate::Statuses status = FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()
                                                         ? ov::test::utils::PassRate::Statuses::SKIPPED
                                                         : ov::test::utils::PassRate::Statuses::CRASHED;
    auto& summary = ov::test::utils::OpSummary::getInstance();
    summary.setDeviceName(targetDevice);
    summary.updateOPsStats(function, status);

    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    functionRefs = ngraph::clone_function(*function);

    std::cout << "VpuOv1LayerTestsCommon::BuildNetworkWithoutCompile" << std::endl;
    BuildNetworkWithoutCompile();
    VpuTestReport& report = VpuTestReport::getInstance();
    const auto& testInfo = testing::UnitTest::GetInstance()->current_test_info();
    report.run(testInfo);

    std::string errorMessage;
    try {
        if (envConfig.IE_NPU_TESTS_RUN_COMPILER) {
            std::cout << "VpuOv1LayerTestsCommon::Compile" << std::endl;
            std::ostringstream ostr;
            ostr << "LoadNetwork Config: ";
            for (const auto& item : configuration) {
                ostr << item.first << "=" << item.second << "; ";
            }
            std::cout << ostr.str() << std::endl;
            SkipBeforeLoad();

            ASSERT_NO_THROW(executableNetwork = getCore()->LoadNetwork(cnnNetwork, targetDevice, configuration));
            report.compiled(testInfo);

            if (envConfig.IE_NPU_TESTS_RUN_EXPORT) {
                std::cout << "VpuOv1LayerTestsCommon::ExportNetwork()" << std::endl;
                ASSERT_NO_THROW(ExportNetwork());
            }
        } else {
            std::cout << "VpuOv1LayerTestsCommon::ImportNetwork()" << std::endl;
            SkipBeforeLoad();
            SkipBeforeImport();
            ImportNetwork();
            report.imported(testInfo);
        }
        GenerateInputs();
        if (envConfig.IE_NPU_TESTS_EXPORT_INPUT) {
            std::cout << "VpuOv1LayerTestsCommon::ExportInput()" << std::endl;
            ExportInput();
        }
        if (envConfig.IE_NPU_TESTS_IMPORT_INPUT) {
            std::cout << "VpuOv1LayerTestsCommon::ImportInput()" << std::endl;
            ImportInput();
        }

        bool runInfer = envConfig.IE_NPU_TESTS_RUN_INFER;
        std::string runInferSkipReason = runInfer ? "-" : "environment variable value";

        const std::string backendName = getBackendName(*core);
        // turn off running infers forcefully for the cases:
        const auto noDevice = backendName.empty();
        if (runInfer && noDevice) {
            GTEST_FATAL_FAILURE_("Inference cannot run: backend is empty (no device)");
        }

        if (runInfer) {
            std::cout << "VpuOv1LayerTestsCommon::Infer() with backend '" << backendName << "'" << std::endl;
            SkipBeforeInfer();
            Infer();
            report.inferred(testInfo);
        } else {
            std::cout << "Skip VpuOv1LayerTestsCommon::Infer() due to: " << runInferSkipReason << std::endl;
        }
        if (envConfig.IE_NPU_TESTS_EXPORT_REF) {
            std::cout << "VpuOv1LayerTestsCommon::ExportReference()" << std::endl;
            ExportReference(CalculateRefs());
        }
        if (envConfig.IE_NPU_TESTS_EXPORT_OUTPUT) {
            std::cout << "VpuOv1LayerTestsCommon::ExportOutput()" << std::endl;
            ExportOutput();
        }
        if (runInfer) {
            std::cout << "VpuOv1LayerTestsCommon::Validate()" << std::endl;
            SkipBeforeValidate();
            Validate();
            report.validated(testInfo);
        } else {
            std::cout << "Skip VpuOv1LayerTestsCommon::Validate() due to: " << runInferSkipReason << std::endl;
        }
        status = ov::test::utils::PassRate::Statuses::PASSED;
    } catch (const VpuSkipTestException& e) {
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

void VpuOv1LayerTestsCommon::setReferenceSoftwareModeMLIR() {
    configuration[ov::intel_vpux::compilation_mode.name()] = "ReferenceSW";
}

void VpuOv1LayerTestsCommon::setDefaultHardwareModeMLIR() {
    configuration[ov::intel_vpux::compilation_mode.name()] = "DefaultHW";
}

void VpuOv1LayerTestsCommon::setPlatformVPU3700() {
    configuration[ov::intel_vpux::platform.name()] = "VPU3700";
}

void VpuOv1LayerTestsCommon::setPlatformVPU3720() {
    configuration[ov::intel_vpux::platform.name()] = "VPU3720";
}

void VpuOv1LayerTestsCommon::setSingleClusterMode() {
    configuration[ov::intel_vpux::dpu_groups.name()] = std::to_string(1);
    configuration[ov::intel_vpux::dma_engines.name()] = std::to_string(1);
}

void VpuOv1LayerTestsCommon::setPerformanceHintLatency() {
    configuration[CONFIG_KEY(PERFORMANCE_HINT)] = "LATENCY";
}

void VpuOv1LayerTestsCommon::useELFCompilerBackend() {
    configuration[ov::intel_vpux::use_elf_compiler_backend.name()] = CONFIG_VALUE(YES);
}

void VpuOv1LayerTestsCommon::TearDown() {
    LayerTestsCommon::TearDown();
    PluginCache::get().reset();
}
}  // namespace LayerTestsUtils
