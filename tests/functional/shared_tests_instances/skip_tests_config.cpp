//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <regex>
#include <string>
#include <vector>

#include <vpux/utils/core/logger.hpp>
#include "common/functions.h"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "vpu_ov1_layer_test.hpp"

class BackendName {
public:
    BackendName() {
        const auto corePtr = PluginCache::get().ie();
        if (corePtr != nullptr) {
            _name = getBackendName(*corePtr);
        } else {
            _log.error("Failed to get IE Core from cache!");
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

    bool isIMD() const {
        return _name == "IMD";
    }

private:
    std::string _name;
    vpux::Logger _log = vpux::Logger::global().nest("BackendName", 0);
};

class AvailableDevices {
public:
    AvailableDevices() {
        const auto corePtr = PluginCache::get().ie();
        if (corePtr != nullptr) {
            _availableDevices = ::getAvailableDevices(*corePtr);
        } else {
            _log.error("Failed to get IE Core from cache!");
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
    vpux::Logger _log = vpux::Logger::global().nest("AvailableDevices", 0);
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
                    _log.info("{0}; Pattern: {1}", entry._comment, pattern);
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
    vpux::Logger _log = vpux::Logger::global().nest("SkipRegistry", 0);
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
                "NPU Plugin doesn't handle DEVICE_ID in QueryNetwork implementation", {
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
        });

        // TODO:
        _skipRegistry.addPatterns(
                "GetExecGraphInfo function is not implemented for VPU plugin", {
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
                "SetConfig function is not implemented for ExecutableNetwork interface (implemented only for vpu plugin)", {
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

        _skipRegistry.addPatterns(
                "This is openvino specific test", {
                ".*ExecutableNetworkBaseTest.canExport.*",
        });

        // TODO:
        _skipRegistry.addPatterns(
                "Issue: E#63469", {
                ".*ConversionLayerTest.*ConvertLike.*",
        });

        _skipRegistry.addPatterns(
                "TensorIterator layer is not supported", {
                ".*ReturnResultNotReadyFromWaitInAsyncModeForTooSmallTimeout.*",
                ".*OVInferRequestDynamicTests.*",
                ".*OVInferenceChaining.*",
        });

        _skipRegistry.addPatterns(
                "Tests with unsupported precision", {
                ".*InferRequestCheckTensorPrecision.*type=boolean.*",
                ".*InferRequestCheckTensorPrecision.*type=bf16.*",
                ".*InferRequestCheckTensorPrecision.*type=f64.*",
                ".*InferRequestCheckTensorPrecision.*type=i4.*",
                ".*InferRequestCheckTensorPrecision.*type=u4.*",
                ".*InferRequestCheckTensorPrecision.*type=u1\\D.*",
                // [Track number: E#97469]
                ".*InferRequestCheckTensorPrecision.*type=i64.*",
        });

        _skipRegistry.addPatterns(!backendName.isZero() || !devices.has3720(),
                "Tests enabled only for L0 NPU3720", {
                // [Track number: E#70764]
                ".*InferRequestCheckTensorPrecision.*",
                ".*InferRequestIOTensorSetPrecisionTest.*",
                ".*VpuxDriverCompilerAdapterDowngradeInterpolate11Test.*",
                ".*VpuxDriverCompilerAdapterInputsOutputsTest.*",
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

        // [Track number: E#89084]
        _skipRegistry.addPatterns(
                "Not expected behavior", {
                R"(.*Auto_Behavior.*InferRequestIOBBlobSetLayoutTest.*layout=ANY.*)",
        });

        // [Track number: E#67741]
        _skipRegistry.addPatterns(
                "Cannot call setShape for Blobs", {
                R"(.*(smoke_Behavior|smoke_Multi_Behavior).*OVInferRequestIOTensorTest.*canInferAfterIOBlobReallocation.*)",
                R"(.*(smoke_Behavior|smoke_Multi_Behavior).*OVInferRequestIOTensorTest.*InferStaticNetworkSetChangedInputTensorThrow.*targetDevice=(VPU_|VPUX_|NPU_|MULTI_configItem=MULTI_DEVICE_PRIORITIES_VPU|MULTI_configItem=MULTI_DEVICE_PRIORITIES_NPU).*)"
        });

        // [Track number: E#67743]
        _skipRegistry.addPatterns(
                "throwOnFail: zeCommandQueueExecuteCommandLists result: ZE_RESULT_ERROR_UNKNOWN, code 0x7ffffffe", {
                ".*smoke_Behavior.*InferRequestIOBBlobTest.*canReallocateExternalBlobViaGet.*",
        });

        // [Track number: E#67747]
        _skipRegistry.addPatterns(
                "AUTOload all devices fail", {
                ".*Auto_Behavior.*InferRequestIOBBlobTest.*canReallocateExternalBlobViaGet.*MULTI_DEVICE_PRIORITIES_VPU_.*",
        });

        // [Track number: E#67749]
        _skipRegistry.addPatterns(
                "Can't loadNetwork without cache for ReadConcatSplitAssign with precision f32", {
                ".*CachingSupportCase_KeemBay.*CompileModelCacheTestBase.*CompareWithRefImpl.*ReadConcatSplitAssign.*",
        });

        // [Tracking number: E#99817]
        _skipRegistry.addPatterns(
                "NPU Plugin currently fails to get a valid output in these test cases", {
                ".*InferRequestIOBBlobTest.canProcessDeallocatedOutputBlobAfterGetAndSetBlob.*",
                ".*InferRequestIOBBlobTestVpux.canProcessDeallocatedOutputBlobAfterGetAndSetBlob.*",
                R"(.*InferRequestIOBBlobTest.secondCallGetInputDoNotReAllocateData/targetDevice=(NPU3720_).*)",
                R"(.*InferRequestIOBBlobTestVpux.secondCallGetInputDoNotReAllocateData/targetDevice=(NPU3720_).*)",
                R"(.*InferRequestIOBBlobTest.secondCallGetOutputDoNotReAllocateData/targetDevice=(NPU3720_).*)",
                R"(.*InferRequestIOBBlobTestVpux.secondCallGetOutputDoNotReAllocateData/targetDevice=(NPU3720_).*)",
                R"(.*InferRequestIOBBlobTest.secondCallGetInputAfterInferSync/targetDevice=(NPU3720_).*)",
                R"(.*InferRequestIOBBlobTestVpux.secondCallGetInputAfterInferSync/targetDevice=(NPU3720_).*)",
                R"(.*InferRequestIOBBlobTest.secondCallGetOutputAfterInferSync/targetDevice=(NPU3720_).*)",
                R"(.*InferRequestIOBBlobTestVpux.secondCallGetOutputAfterInferSync/targetDevice=(NPU3720_).*)",
                ".*OVInferRequestIOTensorTest.InferStaticNetworkSetChangedInputTensorThrow.*",
                ".*OVInferRequestIOTensorTestVpux.InferStaticNetworkSetChangedInputTensorThrow.*",
                R"(.*InferRequestIOBBlobTest.canReallocateExternalBlobViaGet/targetDevice=(NPU3720_)configItem=MULTI_DEVICE_PRIORITIES_NPU.*)",
                R"(.*OVInferRequestIOTensorTestVpux.InferStaticNetworkSetChangedInputTensorThrow/targetDevice=(NPU3720_).*)",
                R"(.*OVInferRequestIOTensorTestVpux.InferStaticNetworkSetChangedInputTensorThrow/targetDevice=(NPU3720_)configItem=MULTI_DEVICE_PRIORITIES_NPU_.*)",
                R"(.*OVInferRequestIOTensorTest.InferStaticNetworkSetChangedInputTensorThrow/targetDevice=(NPU3720_).*)",
                R"(.*OVInferRequestIOTensorTest.InferStaticNetworkSetChangedInputTensorThrow/targetDevice=(NPU3720_)configItem=MULTI_DEVICE_PRIORITIES_NPU_.*)",
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
                R"(.*smoke_Auto_BehaviorTests.*IncorrectConfigTests.CanNotLoadNetworkWithIncorrectConfig.*AUTO_config.*unknown_file_MULTI_DEVICE_PRIORITIES=(VPU_|VPU|VPUX_|VPUX|NPU_|NPU,CPU_).*)"
        });

        // [Track number: E#77756]
        _skipRegistry.addPatterns(
                "OV expects the plugin to not throw any exception on network load, but it actually throws", {
                R"(.*(smoke_Multi_Behavior|smoke_Auto_Behavior).*SetPropLoadNetWorkGetPropTests.*SetPropLoadNetWorkGetProperty.*)"
        });

        // [Track number: E#68776]
        _skipRegistry.addPatterns(
                "Plugin can not perform SetConfig for value like: device=VPU config key=LOG_LEVEL value=0", {
                "smoke_BehaviorTests/DefaultValuesConfigTests.CanSetDefaultValueBackToPlugin.*",
        });

        // [Track number: E#80555]
        _skipRegistry.addPatterns(devices.has3700(),
                "Problems with SplitConcat ngraph function", {
                R"(.*smoke_BehaviorTests/InferRequest(CallbackTests|MultithreadingTests|PerfCountersTest|WaitTests)\..*)",
                R"(.*smoke_BehaviorTests(/|/OV)InferRequestCancellationTests\..*)",
                R"(.*smoke(_|_Auto_)BehaviorTests/InferRequestIOBBlobTest\..*)",
                R"(.*smoke(_|_Multi_)BehaviorTests/OVInferRequestIOTensorTest\..*)",
                R"(.*smoke(_|_Auto_|_Multi_)BehaviorTests/OVInferRequest(CallbackTests|IOTensorSetPrecisionTest|MultithreadingTests)\..*)",
                ".*OVClassNetworkTestP.LoadNetworkActual.*",
                ".*OVClassLoadNetworkTest.LoadNetworkHETEROWithDeviceIDNoThrow.*",
                R"(.*OVHoldersTest\..*)",
                R"(.*OVHoldersTestOnImportedNetwork\..*)",
        });

        // [Track number: E#84510]
        _skipRegistry.addPatterns(
                "Test failed on static shape ([1,3,24,24]): rel_threshold: 0.01, rel_max: 0.225794", {
                "smoke.*PrePostProcess.*PreProcessTestCommon.*NPU3720_HW.Func=scale_mean.*",
        });

        _skipRegistry.addPatterns(
                "Disabled with ticket number", {
                // [Track number: E#48480]
                ".*OVExecutableNetworkBaseTest.*",

                // [Track number: E#63708]
                ".*smoke_BehaviorTests.*InferStaticNetworkSetInputTensor.*",
                ".*smoke_Multi_BehaviorTests.*InferStaticNetworkSetInputTensor.*",

                // [Track number: E#64490]
                 ".*OVClassNetworkTestP.*SetAffinityWithConstantBranches.*"
        });

        // [Tracking number: E#86380]
        _skipRegistry.addPatterns(
                "The output tensor gets freed when the inference request structure's destructor is called. The issue is unrelated to the caching feature.", {
                ".*CacheTestBase.CompareWithRefImpl.*",
        });

        _skipRegistry.addPatterns(
                "Expected: ie->SetConfig(configuration, target_device) throws an exception of type InferenceEngine::Exception. Throws nothing.", {
                // [Tracking number: E#89274]
                ".*AutoBatch.*Behavior.*IncorrectConfigAPITests.SetConfigWithNoExistingKey.*AUTO_BATCH_TIMEOUT.*",
                // [Track number: E#89084]
                ".*AutoBatch.*Behavior.*IncorrectConfigTests.SetConfigWithIncorrectKey.*AUTO_BATCH_TIMEOUT.*",
                ".*AutoBatch.*Behavior.*IncorrectConfigTests.CanNotLoadNetworkWithIncorrectConfig.*AUTO_BATCH_TIMEOUT.*",
        });

        _skipRegistry.addPatterns(
                "Dynamic I/O shapes are being used when running the tests. This feature is not yet supported by the NPU plugin.", {
                ".*SetPreProcessTo.*"
        });

        _skipRegistry.addPatterns(
                "This scenario became invalid upon refactoring the implementation as to use the 2.0 OV API. "
                "The \"set tensor\" operation expects the shape of the new tensor to match the shape of the previous one.", {
                ".*InferRequestIOBBlobSetLayoutTest.*ANY.*"
        });

        _skipRegistry.addPatterns(
                "This scenario became invalid upon refactoring the implementation as to use the 2.0 OV API. "
                "The legacy version structure contains major and minor version attributes, but these fields are not found anymore "
                "in the corresponding 2.0 API structure.", {
                ".*smoke_BehaviorTests/VersionTest.pluginCurrentVersionIsCorrect.*"
        });

        // [Tracking number: E#102428]
        _skipRegistry.addPatterns(
                "Tests throw errors as expected but drivers post-v.1657 will fail to catch them", {
                ".*FailGracefullyTest.*",
                ".*VPUQueryNetworkTestSuite3.*"
        });

        //
        // Conditionally disabled test patterns
        //

        _skipRegistry.addPatterns(devices.count() && !devices.has3720(), "Tests are disabled for all devices except VPU3720",
                                  {
                                          // [Track number: E#49620]
                                          ".*NPU3700(\\.|_)(SW|HW).*",
                                          ".*VPU3720.*",
                                          // [Track number: E#84621]
                                          ".*VpuxDriverCompilerAdapterDowngradeInterpolate11Test.*",
                                          ".*VpuxDriverCompilerAdapterInputsOutputsTest.*",
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
                        ".*VpuxDriverCompilerAdapterInputsOutputsTest.*",

                        // Exception in case of network compilation without devices in system
                        // [Track number: E#30824]
                        ".*OVClassImportExportTestP.*",
                        ".*OVClassLoadNetworkTest.*LoadNetwork.*",
                        // [Track number: E#84621]
                        ".*VpuxDriverCompilerAdapterDowngradeInterpolate11Test.*",
                        ".*VPUQueryNetworkTest.*",
                });

        // [Track number: S#14836]
        _skipRegistry.addPatterns(devices.has3700(),
                "Async tests break on dKMB", {
                ".*ExclusiveAsyncRequests.*",
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

        _skipRegistry.addPatterns(backendName.isZero() && devices.has3700(),
                        "Tests fail on RPL dKMB boards, start_async() fails intermittently",
                        { // [Tracking number: E#90056]
                          ".*OVInferConsistencyTest.*",
                          // [Tracking number: E#92317]
                          ".*OVInferRequestIOTensorTest.*",
                          ".*OVInferRequestMultithreadingTests.*",
                          ".*InferRequestIOBBlobTest.*",
                          ".*OVInferRequestCallbackTests.*",
                          ".*InferRequestMultithreadingTests.*",
                          ".*InferRequestPerfCountersTest.*",
                          ".*InferRequestWaitTests.*",
                          ".*OVInferRequestCancellationTest.*",
                          ".*InferRequestRunTests.*",
                          ".*InferRequestCallbackTests.*",
                          ".*InferRequestCancellationTests.*"});

        _skipRegistry.addPatterns(!(backendName.isZero()), "These tests runs only on LevelZero backend",
                                  {".*InferRequestRunTests.*",
                                   ".*OVClassGetMetricAndPrintNoThrow.*",
                                   ".*IEClassGetMetricAndPrintNoThrow.*",
                                   ".*CompileModelLoadFromFileTestBase.*",
                                   ".*CorrectConfigTests.*"});

        _skipRegistry.addPatterns(!(devices.has3720()), "Runs only on NPU3720 with Level Zero enabled #85493",
                                  {".*InferRequestRunTests.MultipleExecutorStreamsTestsSyncInfers.*"});

        _skipRegistry.addPatterns("Other devices than NPU doesn't allow to set NPU properties with OV1.0 and CACHE_DIR + MLIR is not supported",
                                  {".*smoke_AutoBatch_BehaviorTests/CorrectConfigTests.*"});

        _skipRegistry.addPatterns("OpenVINO issues when using caching mechanism",
                                  {// [Tracking number: C#119359]
                                   ".*smoke_Auto_CachingSupportCase_KeemBay/CompileModelLoadFromFileTestBase.*",
                                   // [Tracking number: C#120240]
                                   ".*smoke_CachingSupportCase_KeemBay/CompileModelLoadFromFileTestBase.*"});

        _skipRegistry.addPatterns(devices.has3700(), "Do not run the tests that require a new CiD version on the 3700 platform",
                                  {
                                   // [Tracking number: E#92279]
                                   ".*CompileModelLoadFromFileTestBase.*",
                                   ".*CorrectConfigTests.CanUseCache.*",
                                   ".*CorrectConfigTests.CanLoadNetworkWithCorrectConfig.*",
                                   ".*VpuxDriverCompilerAdapterDowngradeInterpolate11Test.CheckOpsetVersion.*",
                                   ".*VpuxDriverCompilerAdapterInputsOutputsTest.CheckInOutputs.*",
                                   ".*VpuDriverCompilerAdapterExpectedThrow.CheckWrongGraphExtAndThrow.*"});

        _skipRegistry.addPatterns(backendName.isZero(), "Most ProfilingTest_VPU3700 instances break sporadically, only stable instances are left, #65844", {
                                                ".*precommit_profilingDisabled/ProfilingTest_VPU3700.*",
                                                ".*precommit_profilingDisabled_drv/ProfilingTest_VPU3700.*",
                                                ".*precommit_profilingEnabled_drv/ProfilingTest_VPU3700.*",
                                                });

        _skipRegistry.addPatterns(backendName.isIMD(), "IMD/Simics do not support the tests",
                                  {
                                        // [Tracking number: E#81065]
                                        ".*smoke_VPUXClassPluginProperties.*DEVICE_UUID.*",
                                  });
        _skipRegistry.addPatterns(backendName.isIMD(), "Run long time on IMD/Simics",
                                  {
                                        // [Tracking number: E#85488]
                                        ".*VpuxPreprocessingPrecisionConvertTest.*",
                                  });

        _skipRegistry.addPatterns(!backendName.isZero() || !devices.has3720(),
                "Tests enabled only for L0 VPU3720", {
                // [Track number: E#83423]
                ".*smoke_VariableStateBasic.*",
                // [Track number: E#83708]
                ".*smoke_MemoryLSTMCellTest.*",
        });

        _skipRegistry.addPatterns(!backendName.isZero() || !devices.has3720(),
                "QueryNetwork is only supported by 3720 platform", {
                ".*VPUQueryNetworkTest.*"
        });

        // [Tracking number: E#98601]
        _skipRegistry.addPatterns(backendName.isZero() && devices.has3720(),
                "Test fails when run with Multi_Device_Priorities=NPU,CPU", {
                ".*InferRequestIOBBlobTestVpux.canReallocateExternalBlobViaGet.*MULTI_DEVICE_PRIORITIES_.*CPU.*"
        });

        _skipRegistry.addPatterns(
                devices.count() > 1,
                "Some VPU Plugin metrics require single device to work in auto mode or set particular device",
                {
                        ".*OVClassGetConfigTest.*GetConfigNoThrow.*",
                        ".*OVClassGetConfigTest.*GetConfigHeteroNoThrow.*",
                });

     _skipRegistry.addPatterns(backendName.isZero() && devices.has3720(), "E#93069", {".*InferRequestCheckTensorPrecision.*type=u32.*"});

        return _skipRegistry;
    }();
    // clang-format on

    std::vector<std::string> matchingPatterns;
    const auto currentTestName = getCurrentTestName();
    matchingPatterns.emplace_back(skipRegistry.getMatchingPattern(currentTestName));

    return matchingPatterns;
}
