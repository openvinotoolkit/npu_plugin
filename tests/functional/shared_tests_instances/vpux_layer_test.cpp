//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux_layer_test.hpp"
#include <signal.h>
#include <stdlib.h>
#include <common/utils.hpp>
#include <common_test_utils/common_utils.hpp>
#include <exception>
#include <functional_test_utils/skip_tests_config.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <openvino/runtime/core.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <vpux/utils/IE/config.hpp>
#include <vpux/utils/core/error.hpp>
#include "kmb_test_report.hpp"

namespace VPUXLayerTestsUtils {

const LayerTestsUtils::KmbTestEnvConfig VPUXLayerTestsCommon::envConfig = LayerTestsUtils::KmbTestEnvConfig{};

VPUXLayerTestsCommon::VPUXLayerTestsCommon() {
    IE_ASSERT(core != nullptr);
    _log.setLevel(vpux::LogLevel::Warning);

    if (!envConfig.IE_KMB_TESTS_LOG_LEVEL.empty()) {
        core->set_property(testPlatformTargetDevice, {{CONFIG_KEY(LOG_LEVEL), envConfig.IE_KMB_TESTS_LOG_LEVEL}});

        const auto logLevel = vpux::OptionParser<vpux::LogLevel>::parse(envConfig.IE_KMB_TESTS_LOG_LEVEL);
        _log.setLevel(logLevel);
    }
}

void VPUXLayerTestsCommon::configure_model() {
    const char* DEFAULT_IE_KMB_TESTS_INFERENCE_SHAVES = "16";
    configuration[VPUX_CONFIG_KEY(INFERENCE_SHAVES)] = DEFAULT_IE_KMB_TESTS_INFERENCE_SHAVES;
    if (configuration.count(VPUX_CONFIG_KEY(PLATFORM)) == 0) {
        configuration[VPUX_CONFIG_KEY(PLATFORM)] = envConfig.IE_KMB_TESTS_PLATFORM;
    }

    SubgraphBaseTest::configure_model();
}

void VPUXLayerTestsCommon::useCompilerMLIR() {
    configuration[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);
}

void VPUXLayerTestsCommon::setReferenceSoftwareModeMLIR() {
    configuration[VPUX_CONFIG_KEY(COMPILATION_MODE)] = "ReferenceSW";
}

void VPUXLayerTestsCommon::setDefaultHardwareModeMLIR() {
    configuration[VPUX_CONFIG_KEY(COMPILATION_MODE)] = "DefaultHW";
}

void VPUXLayerTestsCommon::setPlatformVPU3720() {
    configuration[VPUX_CONFIG_KEY(PLATFORM)] = "VPU3720";
}

bool VPUXLayerTestsCommon::isCompilerMLIR() const {
    const auto it = configuration.find(VPUX_CONFIG_KEY(COMPILER_TYPE));
    if (it == configuration.end()) {
        return false;
    }

    return it->second.as<std::string>() == VPUX_CONFIG_VALUE(MLIR);
}

bool VPUXLayerTestsCommon::isPlatformVPU3720() const {
    const auto it = configuration.find(VPUX_CONFIG_KEY(PLATFORM));
    if (it == configuration.end()) {
        return false;
    }

    return it->second.as<std::string>() == "VPU3720";
}

void VPUXLayerTestsCommon::run() {
    ov::test::utils::PassRate::Statuses status = FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()
                                                         ? ov::test::utils::PassRate::Statuses::SKIPPED
                                                         : ov::test::utils::PassRate::Statuses::CRASHED;
    summary.setDeviceName(targetDevice);
    summary.updateOPsStats(function, status);

    SKIP_IF_CURRENT_TEST_IS_DISABLED();

    ASSERT_FALSE(targetStaticShapes.empty()) << "Target Static Shape is empty!";

    // Using KmbTestReport to have a uniform class for both OV test frameworks
    auto& report = LayerTestsUtils::KmbTestReport::getInstance();
    const auto& testInfo = testing::UnitTest::GetInstance()->current_test_info();
    report.run(testInfo);

    std::string errorMessage;
    try {
        printNetworkConfig();
        if (skipBeforeLoadImpl()) {
            return;
        }

        if (envConfig.IE_KMB_TESTS_RUN_COMPILER) {
            std::cout << "*** COMPILE MODEL ***\n";
            compile_model();
            report.compiled(testInfo);
            exportNetwork();
        } else {
            importNetwork();
            report.imported(testInfo);
        }

        for (const auto& targetStaticShapeVec : targetStaticShapes) {
            try {
                if (!inputDynamicShapes.empty()) {
                    ngraph::helpers::resize_function(functionRefs, targetStaticShapeVec);
                }

                generate_inputs(targetStaticShapeVec);
                exportInput();
                importInput();

                if (skipBeforeInferImpl()) {
                    continue;
                }

                std::cout << "*** INFER USING " << getBackendName(*core) << " ***\n";
                infer();
                report.inferred(testInfo);

                if (skipBeforeValidateImpl()) {
                    continue;
                }

                std::cout << "*** VALIDATE ***\n";
                validate();
                report.validated(testInfo);
                exportReference();
                exportOutput();
            } catch (const std::exception& ex) {
                throw std::runtime_error("Incorrect target static shape: " +
                                         CommonTestUtils::vec2str(targetStaticShapeVec) + "\n" + ex.what());
            }
        }
        status = ov::test::utils::PassRate::Statuses::PASSED;
    } catch (const std::exception& ex) {
        status = ov::test::utils::PassRate::Statuses::FAILED;
        errorMessage = ex.what();
    } catch (...) {
        status = ov::test::utils::PassRate::Statuses::FAILED;
        errorMessage = "Unknown failure occurred.";
    }

    summary.updateOPsStats(function, status);
    if (status != ov::test::utils::PassRate::Statuses::PASSED) {
        _log.error("Test has failed: {0}", errorMessage.c_str());
        GTEST_FATAL_FAILURE_(errorMessage.c_str());
    }
}

bool VPUXLayerTestsCommon::skipBeforeLoadImpl() {
    if (auto skipMessage = SkipBeforeLoad()) {
        _log.warning("Compilation skipped: {0}", skipMessage.getValue());
        return true;
    }

    return false;
}

bool VPUXLayerTestsCommon::skipBeforeInferImpl() {
    const auto backendName = getBackendName(*core);

    if (backendName.empty()) {
        _log.warning("Inference skipped: backend is empty (no device)");
        return true;
    }

    // [Track number: E#20335]
    // Disabling inference for layer tests on emulator device due to segfault
    if (backendName == "EMULATOR") {
        _log.warning("Inference skipped: backend is EMULATOR");
        return true;
    }

    if (auto skipMessage = SkipBeforeInfer()) {
        _log.warning("Inference skipped: {0}", skipMessage.getValue());
        return true;
    }

    return false;
}

bool VPUXLayerTestsCommon::skipBeforeValidateImpl() {
    if (auto skipMessage = SkipBeforeValidate()) {
        _log.warning("Validation skipped: {0}", skipMessage.getValue());
        return true;
    }

    return false;
}

SkipMessage VPUXLayerTestsCommon::SkipBeforeLoad() {
    return {};
}
SkipMessage VPUXLayerTestsCommon::SkipBeforeInfer() {
    return {};
}
SkipMessage VPUXLayerTestsCommon::SkipBeforeValidate() {
    return {};
}

void VPUXLayerTestsCommon::importNetwork() {
    // [Track number: E#24517]
    _log.warning("VPUXLayerTestsCommon::importNetwork() is not implemented");
}

void VPUXLayerTestsCommon::exportNetwork() {
    // [Track number: E#24517]
    if (envConfig.IE_KMB_TESTS_RUN_EXPORT) {
        _log.warning("VPUXLayerTestsCommon::exportNetwork() is not implemented");
    }
}

void VPUXLayerTestsCommon::importInput() {
    // [Track number: E#24517]
    if (envConfig.IE_KMB_TESTS_IMPORT_INPUT) {
        _log.warning("VPUXLayerTestsCommon::importInput() is not implemented");
    }
}

void VPUXLayerTestsCommon::exportInput() {
    // [Track number: E#24517]
    if (envConfig.IE_KMB_TESTS_EXPORT_INPUT) {
        _log.warning("VPUXLayerTestsCommon::exportInput() is not implemented");
    }
}

void VPUXLayerTestsCommon::exportOutput() {
    // [Track number: E#24517]
    if (envConfig.IE_KMB_TESTS_EXPORT_OUTPUT) {
        _log.warning("VPUXLayerTestsCommon::exportOutput() is not implemented");
    }
}

void VPUXLayerTestsCommon::exportReference() {
    // [Track number: E#24517]
    if (envConfig.IE_KMB_TESTS_EXPORT_REF) {
        _log.warning("VPUXLayerTestsCommon::exportReference() is not implemented");
    }
}

void VPUXLayerTestsCommon::printNetworkConfig() const {
    std::ostringstream ostr;
    for (const auto& item : configuration) {
        ostr << item.first << "=";
        item.second.print(ostr);
        ostr << "; ";
    }
    std::cout << "LoadNetwork Config: " << ostr.str() << '\n';
}

}  // namespace VPUXLayerTestsUtils
