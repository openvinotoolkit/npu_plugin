//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <regex>
#include <string>
#include <vector>

#include "common/functions.h"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "kmb_layer_test.hpp"

class BackendName {
public:
    BackendName() {
        const auto corePtr = PluginCache::get().ie();
        if (corePtr != nullptr) {
            _name = getBackendName(*corePtr);
        } else {
            std::cout << "Failed to get IE Core!" << std::endl;
        }
    }

    std::string getName() const {
        return _name;
    }

    bool isEmpty() const noexcept {
        return _name.empty();
    }

    bool isZero() const {
        return _name == "LEVEL0";
    }

    bool isVpual() const {
        return _name == "VPUAL";
    }

    bool isEmulator() const {
        return _name == "EMULATOR";
    }

    bool isIMD() const {
        return _name == "IMD";
    }

private:
    std::string _name;
};

class AvailableDevices {
public:
    AvailableDevices() {
        const auto corePtr = PluginCache::get().ie();
        if (corePtr != nullptr) {
            _availableDevices = ::getAvailableDevices(*corePtr);
        } else {
            std::cout << "Failed to get IE Core!" << std::endl;
        }
    }

    const auto& getAvailableDevices() const {
        return _availableDevices;
    }

    auto count() const {
        return _availableDevices.size();
    }

    bool has3720() const {
        return std::any_of(_availableDevices.begin(), _availableDevices.end(), [](const std::string& deviceName) {
            return deviceName.find("3720") != std::string::npos;
        });
    }

    bool has3700() const {
        return std::any_of(_availableDevices.begin(), _availableDevices.end(), [](const std::string& deviceName) {
            return deviceName.find("3700") != std::string::npos;
        });
    }

private:
    std::vector<std::string> _availableDevices;
};

class Platform {
public:
    bool isARM() const noexcept {
#if defined(__arm__) || defined(__aarch64__)
        return true;
#else
        return false;
#endif
    }

    bool isX86() const noexcept {
#if defined(__x86_64) || defined(__x86_64__)
        return true;
#else
        return false;
#endif
    }
};

class SkipRegistry {
public:
    void addPatterns(std::string&& comment, std::vector<std::string>&& patternsToSkip) {
        _registry.emplace_back(std::move(comment), std::move(patternsToSkip));
    }

    void addPatterns(bool conditionFlag, std::string&& comment, std::vector<std::string>&& patternsToSkip) {
        if (conditionFlag) {
            addPatterns(std::move(comment), std::move(patternsToSkip));
        }
    }

    /** Searches for the skip pattern to which passed test name matches.
     * Prints the message onto console if pattern is found and the test is to be skipped
     *
     * @param testName name of the current test being matched against skipping
     * @return Suitable skip pattern or empty string if none
     */
    std::string getMatchingPattern(const std::string& testName) const {
        for (const auto& entry : _registry) {
            for (const auto& pattern : entry._patterns) {
                std::regex re(pattern);
                if (std::regex_match(testName, re)) {
                    std::cout << entry._comment << "; Pattern: " << pattern << std::endl;
                    return pattern;
                }
            }
        }

        return std::string{};
    }

private:
    struct Entry {
        Entry(std::string&& comment, std::vector<std::string>&& patterns)
                : _comment{std::move(comment)}, _patterns{std::move(patterns)} {
        }

        std::string _comment;
        std::vector<std::string> _patterns;
    };

    std::vector<Entry> _registry;
};

std::string getCurrentTestName() {
    const auto* currentTestInfo = ::testing::UnitTest::GetInstance()->current_test_info();
    const auto currentTestName = currentTestInfo->test_case_name() + std::string(".") + currentTestInfo->name();
    return currentTestName;
}

std::vector<std::string> disabledTestPatterns() {
    // Initialize skip registry
    static const auto skipRegistry = []() {
        SkipRegistry _skipRegistry;

        const BackendName backendName;
        const AvailableDevices devices;
        const Platform platform;

        // clang-format off

        //
        //  Disabled test patterns
        //
        // TODO
        _skipRegistry.addPatterns(
                "Tests break due to starting infer on IA side", {
                ".*CorrectConfigAPITests.*",
        });

        _skipRegistry.addPatterns(
                "ARM CPU Plugin is not available on Yocto", {
                ".*IEClassLoadNetworkTest.*HETERO.*",
                ".*IEClassLoadNetworkTest.*MULTI.*",
        });

        // TODO
        // [Track number: E#30810]
        _skipRegistry.addPatterns(
                "Hetero plugin doesn't throw an exception in case of big device ID", {
                ".*OVClassLoadNetworkTest.*LoadNetworkHETEROWithBigDeviceIDThrows.*",
        });

        // TODO
        // [Track number: E#30815]
        _skipRegistry.addPatterns(
                "VPUX Plugin doesn't handle DEVICE_ID in QueryNetwork implementation", {
                ".*OVClassQueryNetworkTest.*",
        });

        // [Track number: E#12774]
        _skipRegistry.addPatterns(
                "Cannot detect vpu platform when it's not passed; Skip tests on Yocto which passes device without platform", {
                ".*IEClassLoadNetworkTest.LoadNetworkWithDeviceIDNoThrow.*",
                ".*IEClassLoadNetworkTest.LoadNetworkWithBigDeviceIDThrows.*",
                ".*IEClassLoadNetworkTest.LoadNetworkWithInvalidDeviceIDThrows.*",
        });

        // [Track number: E#28335]
        _skipRegistry.addPatterns(
                "Disabled test E#28335", {
                ".*smoke_LoadNetworkToDefaultDeviceNoThrow.*",
        });

        // [Track number: E#32241]
        _skipRegistry.addPatterns(
                "Disabled test E#28335", {
                ".*LoadNetwork.*CheckDeviceInBlob.*",
        });

        // [Track number: S#27343]
        _skipRegistry.addPatterns(
                "double free detected", {
                ".*InferConfigInTests\\.CanInferWithConfig.*",
                ".*InferConfigTests\\.withoutExclusiveAsyncRequests.*",
                ".*InferConfigTests\\.canSetExclusiveAsyncRequests.*",
        });

        // TODO:
        _skipRegistry.addPatterns(
                "GetExecGraphInfo function is not implemented for VPUX plugin", {
                ".*checkGetExecGraphInfoIsNotNullptr.*",
                ".*CanCreateTwoExeNetworksAndCheckFunction.*",
                ".*CheckExecGraphInfo.*",
                ".*canLoadCorrectNetworkToGetExecutable.*",
        });

        // [Track number: E#31074]
        _skipRegistry.addPatterns(
                "Disabled test E#28335", {
                ".*checkInferTime.*",
                ".*OVExecGraphImportExportTest.*",
        });

        _skipRegistry.addPatterns(
                "Test uses legacy OpenVINO 1.0 API, no need to support it", {
                ".*ExecutableNetworkBaseTest.checkGetMetric.*",
        });

        // TODO:
        _skipRegistry.addPatterns(
                "SetConfig function is not implemented for ExecutableNetwork interface (implemented only for vpux plugin)", {
                ".*ExecutableNetworkBaseTest.canSetConfigToExecNet.*",
                ".*ExecutableNetworkBaseTest.canSetConfigToExecNetAndCheckConfigAndCheck.*",
                ".*CanSetConfigToExecNet.*",
        });

        // TODO
        // [Track number: E#30822]
        _skipRegistry.addPatterns(
                "Exception 'Not implemented'", {
                ".*OVClassNetworkTestP.*LoadNetworkCreateDefaultExecGraphResult.*",
        });

        // TODO: [Track number: S#14836]
        _skipRegistry.addPatterns(
                "Async tests break on dKMB", {
                ".*ExclusiveAsyncRequests.*",
        });

        _skipRegistry.addPatterns(
                "This is openvino specific test", {
                ".*ExecutableNetworkBaseTest.canExport.*",
        });

        // TODO:
        _skipRegistry.addPatterns(
                "Issue: E#63469", {
                ".*VPUXConversionLayerTest.*ConvertLike.*",
        });

        _skipRegistry.addPatterns(
                "TensorIterator layer is not supported", {
                ".*ReturnResultNotReadyFromWaitInAsyncModeForTooSmallTimeout.*",
                ".*OVInferRequestDynamicTests.*",
                ".*OVInferenceChaining.*",
        });

        // Current OV logic with FULL_DEVICE_NAME metric differs from VPUX Plugin
        _skipRegistry.addPatterns(
                "Plugin throws an exception in case of absence VPUX devices in system", {
                ".*IEClassGetMetricTest_FULL_DEVICE_NAME.*GetMetricAndPrintNoThrow.*",
        });

        _skipRegistry.addPatterns(
                "Tests with unsupported precision", {
                ".*InferRequestCheckTensorPrecision.*type=boolean.*",
                ".*InferRequestCheckTensorPrecision.*type=bf16.*",
                ".*InferRequestCheckTensorPrecision.*type=f64.*",
                ".*InferRequestCheckTensorPrecision.*type=i4.*",
                ".*InferRequestCheckTensorPrecision.*type=i16.*",
                ".*InferRequestCheckTensorPrecision.*type=i64.*",
                ".*InferRequestCheckTensorPrecision.*type=u1.*",
                ".*InferRequestCheckTensorPrecision.*type=u4.*",
                ".*InferRequestCheckTensorPrecision.*type=u16.*",
                ".*InferRequestCheckTensorPrecision.*type=u32.*",
                ".*InferRequestCheckTensorPrecision.*type=u64.*",
        });

        // TODO
        // [Track number: E#32075]
        _skipRegistry.addPatterns(
                "Exception during loading to the device", {
                ".*OVClassLoadNetworkTest.*LoadNetworkHETEROwithMULTINoThrow.*",
                ".*OVClassLoadNetworkTest.*LoadNetworkMULTIwithHETERONoThrow.*",
        });

        _skipRegistry.addPatterns(
                "compiler: Unsupported arch kind: VPUX311X", {
                ".*CompilationForSpecificPlatform.*(3800|3900).*",
        });

        _skipRegistry.addPatterns(
                "Not expected behavior, same as for gpu", {
                R"(.*Behavior.*InferRequestIOBBlobSetLayoutTest.*layout=(95|SCALAR|OIHW).*)",
                R"(.*Behavior.*InferRequestIOBBlobSetLayoutTest.*CanSetInBlobWithDifferentLayouts.*layout=NHWC.*)",
                R"(.*Behavior.*InferRequestIOBBlobSetLayoutTest.*CanSetOutBlobWithDifferentLayouts.*layout=(CN|HW).*)",
        });

        // [Track number: E#67741]
        _skipRegistry.addPatterns(
                "Cannot call setShape for Blobs", {
                R"(.*(smoke_Behavior|smoke_Multi_Behavior).*OVInferRequestIOTensorTest.*canInferAfterIOBlobReallocation.*)",
                R"(.*(smoke_Behavior|smoke_Multi_Behavior).*OVInferRequestIOTensorTest.*InferStaticNetworkSetChangedInputTensorThrow.*targetDevice=(VPUX_|MULTI_configItem=MULTI_DEVICE_PRIORITIES_VPUX).*)"
        });

        // [Track number: E#67743]
        _skipRegistry.addPatterns(
                "throwOnFail: zeCommandQueueExecuteCommandLists result: ZE_RESULT_ERROR_UNKNOWN, code 0x7ffffffe", {
                ".*smoke_Behavior.*InferRequestIOBBlobTest.*canReallocateExternalBlobViaGet.*",
        });

        // [Track number: E#67747]
        _skipRegistry.addPatterns(
                "AUTOload all devices fail", {
                ".*Auto_Behavior.*InferRequestIOBBlobTest.*canReallocateExternalBlobViaGet.*MULTI_DEVICE_PRIORITIES_VPUX_.*",
        });

        // [Track number: E#67749]
        _skipRegistry.addPatterns(
                "Can't loadNetwork without cache for ReadConcatSplitAssign with precision f32", {
                ".*CachingSupportCase_KeemBay.*CompileModelCacheTestBase.*CompareWithRefImpl.*ReadConcatSplitAssign.*",
        });

        // [Track number: E#73523]
        _skipRegistry.addPatterns(
                "VPUX Plugin currently fails to get a valid output in these test cases", {
                "smoke_precommit_VPU3720_BehaviorTestsPreprocess/InferRequestPreprocessTest.SetMeanImagePreProcessGetBlob.*",
                "smoke_precommit_VPU3720_BehaviorTestsPreprocess/InferRequestPreprocessTest.SetMeanImagePreProcessSetBlob.*",
                "smoke_precommit_VPU3720_BehaviorTestsPreprocess/InferRequestPreprocessTest.SetMeanValuePreProcessGetBlob.*",
                "smoke_precommit_VPU3720_BehaviorTestsPreprocess/InferRequestPreprocessTest.SetMeanValuePreProcessSetBlob.*",
                "smoke_precommit_VPU3720_BehaviorTestsPreprocess/InferRequestPreprocessTest.SetScalePreProcessGetBlob.*",
                "smoke_precommit_VPU3720_BehaviorTestsPreprocess/InferRequestPreprocessTest.SetScalePreProcessSetBlob.*",
        });


        // [Track number: E#68774]
        _skipRegistry.addPatterns(
                "OV requires the plugin to throw when value of DEVICE_ID is unrecognized, but plugin does not throw", {
                "smoke_BehaviorTests.*IncorrectConfigTests.SetConfigWithIncorrectKey.*(SOME_DEVICE_ID|DEVICE_UNKNOWN).*",
                "smoke_BehaviorTests.*IncorrectConfigTests.SetConfigWithNoExistingKey.*SOME_DEVICE_ID.*",
                "smoke_BehaviorTests.*IncorrectConfigAPITests.SetConfigWithNoExistingKey.*(SOME_DEVICE_ID|DEVICE_UNKNOWN).*",
        });

        // [Track number: E#77755]
        _skipRegistry.addPatterns(
                "OV requires the plugin to throw on network load when config file is incorrect, but plugin does not throw", {
                R"(.*smoke_Auto_BehaviorTests.*IncorrectConfigTests.CanNotLoadNetworkWithIncorrectConfig.*AUTO_config.*unknown_file_MULTI_DEVICE_PRIORITIES=(VPUX_|VPUX,CPU_).*)"
        });

        // [Track number: E#77756]
        _skipRegistry.addPatterns(
                "OV expects the plugin to not throw any exception on network load, but it actually throws", {
                R"(.*(smoke_Multi_Behavior|smoke_Auto_Behavior).*SetPropLoadNetWorkGetPropTests.*SetPropLoadNetWorkGetProperty.*)"
        });

        // [Track number: E#68776]
        _skipRegistry.addPatterns(
                "Plugin can not perform SetConfig for value like: device=VPUX config key=LOG_LEVEL value=0", {
                "smoke_BehaviorTests/DefaultValuesConfigTests.CanSetDefaultValueBackToPlugin.*",
        });

        _skipRegistry.addPatterns(
                "Disabled with ticket number", {
                // [Track number: E#48480]
                ".*OVExecutableNetworkBaseTest.*",
                ".*OVInferRequestCheckTensorPrecision.*",

                // [Track number: E#63708]
                ".*smoke_BehaviorTests.*InferStaticNetworkSetInputTensor.*",
                ".*smoke_Multi_BehaviorTests.*InferStaticNetworkSetInputTensor.*",

                // [Track number: E#64490]
                 ".*OVClassNetworkTestP.*SetAffinityWithConstantBranches.*"
        });

        // [Track number: E#73598]
        _skipRegistry.addPatterns(
                "The test tries to concatenate an extra .0123 after the typical VPUX.0123 name, and it fails currently", {
                "smoke_InterfaceTests.TestEngineClassGetMetric*",
        });

        //
        // Conditionally disabled test patterns
        //

        _skipRegistry.addPatterns(devices.count() && !devices.has3720(), "Tests are disabled for all devices except VPU3720",
                                  {
                                          // [Track number: E#49620]
                                          ".*VPU3700(\\.|_)(SW|HW).*",
                                          ".*VPU3720.*",
                                          // [Track number: E#84621]
                                          ".*VpuxDriverCompilerAdapterDowngradeInterpolate11Test.*",
                                  });

        _skipRegistry.addPatterns(
                backendName.isEmpty(), "Disabled for when backend is empty (i.e., no device)",
                {
                        // Cannot run InferRequest tests without a device to infer to
                        ".*InferRequest.*",
                        ".*OVInferRequest.*",
                        ".*OVInferenceChaining.*",
                        ".*ExecutableNetworkBaseTest.*",
                        ".*OVExecutableNetworkBaseTest.*",
                        ".*ExecNetSetPrecision.*",
                        ".*SetBlobTest.*",
                        ".*InferRequestCallbackTests.*",
                        ".*PrePostProcessTest.*",
                        ".*PreprocessingPrecisionConvertTest.*",
                        ".*SetPreProcessToInputInfo.*",
                        ".*InferRequestPreprocess.*",
                        ".*HoldersTestOnImportedNetwork.*",
                        ".*HoldersTest.Orders.*",
                        ".*HoldersTestImportNetwork.Orders.*",

                        // Cannot compile network without explicit specifying of the platform in case of no devices
                        ".*OVExecGraphImportExportTest.*",
                        ".*OVHoldersTest.*",
                        ".*OVClassExecutableNetworkGetMetricTest.*",
                        ".*OVClassExecutableNetworkGetConfigTest.*",
                        ".*OVClassNetworkTestP.*SetAffinityWithConstantBranches.*",
                        ".*OVClassNetworkTestP.*SetAffinityWithKSO.*",
                        ".*OVClassNetworkTestP.*LoadNetwork.*",
                        ".*FailGracefullyTest.*",

                        // Exception in case of network compilation without devices in system
                        // [Track number: E#30824]
                        ".*OVClassImportExportTestP.*",
                        ".*OVClassLoadNetworkTest.*LoadNetwork.*",
                        // [Track number: E#84621]
                        ".*VpuxDriverCompilerAdapterDowngradeInterpolate11Test.*",
                });

        _skipRegistry.addPatterns(backendName.isZero() && devices.has3700(),
                                  "TensorIterator layer is not supported by dKMB platform",
                                  {
                                          ".*SetBlobTest.*",
                                  });

        _skipRegistry.addPatterns(backendName.isZero() && devices.has3700(), "Abs layer is not supported by dKMB platform",
                                  {".*PrePostProcessTest.*"});

        _skipRegistry.addPatterns(backendName.isZero() && devices.has3700(), "Convert layer is not supported by dKMB platform",
                                 {".*PreprocessingPrecisionConvertTest.*", ".*InferRequestPreprocess.*"});

        _skipRegistry.addPatterns(backendName.isZero(), "Most KmbProfilingTest instances break sporadically, only stable instances are left, #65844", {
                                                ".*precommit_profilingDisabled/KmbProfilingTest.*",
                                                ".*precommit_profilingDisabled_drv/KmbProfilingTest.*",
                                                ".*precommit_profilingNonMatchedName_drv/KmbProfilingTest.*",
                                                ".*precommit_profilingMatchedName_drv/KmbProfilingTest.*",
                                                });

        _skipRegistry.addPatterns(backendName.isIMD(), "IMD/Simics do not support the tests",
                                  {
                                        // [Tracking number: E#81065]
                                        ".*smoke_VPUXClassPluginPropertiesTest/VPUXClassPluginPropertiesTestSuite2.CanNotSetImmutableProperty.*",
                                  });

        _skipRegistry.addPatterns(platform.isARM(), "CumSum layer is not supported by ARM platform",
                                  {
                                          ".*SetBlobTest.CompareWithRefs.*",
                                  });

        // TODO: [Track number: E#26428]
        _skipRegistry.addPatterns(platform.isARM(), "LoadNetwork throws an exception",
                                  {
                                          ".*VPUXGatherLayerTest_VPU3700.HW/.*",
                                  });

        _skipRegistry.addPatterns(!backendName.isZero() || !devices.has3720(),
                "Tests enabled only for L0 VPU3720", {
                // [Track number: E#83423]
                ".*smoke_VariableStateBasic.*",
                // [Track number: E#83708]
                ".*smoke_MemoryLSTMCellTest.*",
        });

        _skipRegistry.addPatterns(
                devices.count() > 1,
                "Some VPUX Plugin metrics require single device to work in auto mode or set particular device",
                {
                        ".*OVClassGetConfigTest.*GetConfigNoThrow.*",
                        ".*OVClassGetConfigTest.*GetConfigHeteroNoThrow.*",
                });

#ifdef __unix__
        _skipRegistry.addPatterns(backendName.isZero() && devices.has3720(), "E#77822", {".*CTCGreedyDecoder.*"});
        _skipRegistry.addPatterns(backendName.isZero() && devices.has3720(), "E#78232", {".*smoke_Tile3720_tiling_2.*"});
        _skipRegistry.addPatterns(backendName.isZero() && devices.has3720(), "E#78232", {".*smoke_Tile3720_tiling_3.*"});
#endif

        return _skipRegistry;
    }();
    // clang-format on

    std::vector<std::string> matchingPatterns;
    const auto currentTestName = getCurrentTestName();
    matchingPatterns.emplace_back(skipRegistry.getMatchingPattern(currentTestName));

    return matchingPatterns;
}
