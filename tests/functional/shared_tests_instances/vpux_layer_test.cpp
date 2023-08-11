//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include "vpux_layer_test.hpp"
#include <gtest/internal/gtest-internal.h>
#include <openvino/runtime/core.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <sstream>
#include <vpux/utils/IE/config.hpp>
#include <vpux_private_config.hpp>

#include "kmb_test_report.hpp"

VPUXLayerTest::VPUXLayerTest() {
    VPUX_THROW_UNLESS(core != nullptr, "ov::Core instance is null");

    _log.setName("VPUXTest");
    _log.setLevel(vpux::LogLevel::Info);

    if (auto var = std::getenv("IE_VPUX_TESTS_LOG_LEVEL")) {
        const auto logLevel = vpux::OptionParser<vpux::LogLevel>::parse(var);
        _log.setLevel(logLevel);
    }
}

void VPUXLayerTest::run(VPUXPlatform platform) {
    setPlatform(platform);
    run();
}

void VPUXLayerTest::run() {
    summary.setDeviceName(targetDevice);

    if (FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {
        summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::SKIPPED);
        GTEST_SKIP_("Disabled test due to configuration");
    }

    summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::CRASHED);

    ASSERT_FALSE(targetStaticShapes.empty()) << "Target Static Shape is empty!";

    auto crashHandler = std::make_unique<CommonTestUtils::CrashHandler>();

#ifdef _WIN32
    switch (setjmp(CommonTestUtils::env)) {
#else
    switch (sigsetjmp(CommonTestUtils::env, 1)) {
#endif
    case CommonTestUtils::JMP_STATUS::ok:
        crashHandler->StartTimer();
        if (auto errorMessage = runTest()) {
            _log.error("Test has failed: {0}", errorMessage->c_str());
            GTEST_FATAL_FAILURE_(errorMessage->c_str());
            summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::FAILED);
        } else {
            summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::PASSED);
        }
        break;
    case CommonTestUtils::JMP_STATUS::anyError:
        GTEST_FATAL_FAILURE_("Crash happened");
        break;
    case CommonTestUtils::JMP_STATUS::alarmErr:
        summary.updateOPsStats(function, ov::test::utils::PassRate::Statuses::HANGED);
        GTEST_FATAL_FAILURE_("Application hanged");
        break;
    default:
        GTEST_FATAL_FAILURE_("Test failed: Unsupported failure type");
    }
}

VPUXLayerTest::ErrorMessage VPUXLayerTest::runTest() {
    try {
        auto& report = LayerTestsUtils::KmbTestReport::getInstance();
        const auto testInfo = testing::UnitTest::GetInstance()->current_test_info();
        report.run(testInfo);

        printNetworkConfig();

        if (skipCompilationImpl()) {
            return std::nullopt;
        }

        _log.debug("Compile model");
        compile_model();
        report.compiled(testInfo);

        for (const auto& targetStaticShapeVec : targetStaticShapes) {
            try {
                if (!inputDynamicShapes.empty()) {
                    ngraph::helpers::resize_function(functionRefs, targetStaticShapeVec);
                }
                generate_inputs(targetStaticShapeVec);
            } catch (const std::exception& ex) {
                return ErrorMessage{"Impossible to reshape ov::Model using the shape: " +
                                    CommonTestUtils::vec2str(targetStaticShapeVec) + " " + ex.what()};
            }

            try {
                if (skipInferenceImpl()) {
                    continue;
                }

                // [Track number: C#104172]
                // The infer() function is called inside validate() -> get_plugin_outputs() function
                _log.info("Infer using '{0}' backend & validate", getBackendName(*core));
                validate();
                report.inferred(testInfo);
                report.validated(testInfo);
            } catch (const std::exception& ex) {
                return ErrorMessage{"Test failed on static shape: " + CommonTestUtils::vec2str(targetStaticShapeVec) +
                                    "\n" + ex.what()};
            }
        }
    } catch (const std::exception& ex) {
        return ErrorMessage{ex.what()};
    } catch (...) {
        return ErrorMessage{"Unknown failure occurred."};
    }

    return std::nullopt;
}

void VPUXLayerTest::setSkipCompilationCallback(SkipCallback skipCallback) {
    skipCompilationCallback = skipCallback;
}

void VPUXLayerTest::setSkipInferenceCallback(SkipCallback skipCallback) {
    skipInferenceCallback = skipCallback;
}

bool VPUXLayerTest::skipCompilationImpl() {
    if (skipCompilationCallback != nullptr) {
        std::stringstream skipStream;
        skipCompilationCallback(skipStream);

        const auto skipMessage = skipStream.str();
        if (!skipMessage.empty()) {
            _log.warning("Compilation skipped: {0}", skipMessage);
            return true;
        }
    }

    return false;
}

bool VPUXLayerTest::skipInferenceImpl() {
    const auto backendName = getBackendName(*core);

    if (backendName.empty()) {
        _log.warning("Inference skipped: backend is empty (no device)");
        return true;
    }

    if (skipInferenceCallback != nullptr) {
        std::stringstream skipStream;
        skipInferenceCallback(skipStream);

        const auto skipMessage = skipStream.str();
        if (!skipMessage.empty()) {
            _log.warning("Inference skipped: {0}", skipStream.str());
            return true;
        }
    }

    return false;
}

void VPUXLayerTest::printNetworkConfig() const {
    std::ostringstream ostr;
    for (const auto& item : configuration) {
        ostr << item.first << "=";
        item.second.print(ostr);
        ostr << "; ";
    }
    _log.info("VPUX Plugin config: {0}", ostr.str());
}

void VPUXLayerTest::setPlatform(VPUXPlatform platform) {
    const auto platformString = [&]() {
        switch (platform) {
        case VPUXPlatform::VPU3700:
            return "3700";
        case VPUXPlatform::VPU3720:
            return "3720";
        default:
            VPUX_THROW("Unsupported platform provided");
            return "None";
        }
    }();

    // [Track number: E#70404]
    // Multiple different ways of setting the platform
    configuration[VPUX_CONFIG_KEY(PLATFORM)] = platformString;
    configuration[CONFIG_KEY(DEVICE_ID)] = platformString;
}

void VPUXLayerTest::setReferenceSoftwareMode() {
    configuration[VPUX_CONFIG_KEY(COMPILATION_MODE)] = "ReferenceSW";
}

void VPUXLayerTest::setDefaultHardwareMode() {
    configuration[VPUX_CONFIG_KEY(COMPILATION_MODE)] = "DefaultHW";
}

bool VPUXLayerTest::isReferenceSoftwareMode() const {
    const auto compilationMode = configuration.at(VPUX_CONFIG_KEY(COMPILATION_MODE)).as<std::string>();
    return compilationMode == "ReferenceSW";
}

bool VPUXLayerTest::isDefaultHardwareMode() const {
    const auto compilationMode = configuration.at(VPUX_CONFIG_KEY(COMPILATION_MODE)).as<std::string>();
    return compilationMode == "DefaultHW";
}

void VPUXLayerTest::setSingleClusterMode() {
    configuration[VPUX_CONFIG_KEY(DPU_GROUPS)] = "1";
    configuration[VPUX_CONFIG_KEY(DMA_ENGINES)] = "1";
}

void VPUXLayerTest::setPerformanceHintLatency() {
    configuration[CONFIG_KEY(PERFORMANCE_HINT)] = "LATENCY";
}

void VPUXLayerTest::useELFCompilerBackend() {
    configuration[VPUX_CONFIG_KEY(USE_ELF_COMPILER_BACKEND)] = CONFIG_VALUE(YES);
}
