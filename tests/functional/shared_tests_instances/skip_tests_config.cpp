//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>
#include <regex>

#include "functional_test_utils/plugin_cache.hpp"
#include "common_test_utils/common_utils.hpp"
#include "kmb_layer_test.hpp"
#include "common/functions.h"


class BackendName {
public:
    BackendName() {
        const auto corePtr = PluginCache::get().ie();
        if (corePtr != nullptr) {
            _name = getBackendName(*corePtr);
            _availableDevices = ::getAvailableDevices(*corePtr);
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

    bool isHddl2() const {
        return _name == "HDDL2";
    }

    bool isEmulator() const {
        return _name == "EMULATOR";
    }

    std::vector<std::string> getAvailableDevices() const {
        return _availableDevices;
    }

private:
    std::string _name;
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
    void addPatterns(std::vector<std::string>&& patternsToSkip) {
        addPatterns(true, // unconditionally disabled
                    std::string{"The test is disabled!"},
                    std::move(patternsToSkip)
        );
    }

    void addPatterns(bool conditionFlag, 
                     std::string&& comment,
                     std::vector<std::string>&& patternsToSkip) {
        if (conditionFlag) {
            _registry.emplace_back(std::move(comment),
                                   std::move(patternsToSkip)
            );
        }
    }

    std::string getMatchingPattern(const std::string& testName) const
    {
        for (const auto& entry : _registry) {
            for (const auto& pattern : entry._patterns) {
                std::regex re(pattern);
                if (std::regex_match(testName, re)) {
                    std::cout << entry._comment << std::endl;
                    return pattern;
                }
            }
        }

        return std::string{};
    }

private:
    struct Entry {
        Entry(std::string&& comment, std::vector<std::string>&& patterns) :
              _comment{std::move(comment)}
            , _patterns{std::move(patterns)} { }
            
        std::string _comment;
        std::vector<std::string> _patterns;
    };

    std::vector<Entry> _registry;
};

std::string getCurrentTestName() {
    const auto* currentTestInfo = ::testing::UnitTest::GetInstance()->current_test_info();
    const auto currentTestName = currentTestInfo->test_case_name()
                                + std::string(".") + currentTestInfo->name();    
    return currentTestName;
}

std::vector<std::string> disabledTestPatterns() {
    // Initialize skip registry
    static const auto skipRegistry = []() {
        SkipRegistry      _skipRegistry;
        const BackendName backendName;
        const Platform    platform;

        //
        //  Disabled test patterns
        //

        _skipRegistry.addPatterns({
            // TODO Tests failed due to starting infer on IA side
            ".*CorrectConfigAPITests.*",

            // ARM CPU Plugin is not available on Yocto
            ".*IEClassLoadNetworkTest.*HETERO.*",
            ".*IEClassLoadNetworkTest.*MULTI.*",

            // TODO Hetero plugin doesn't throw an exception in case of big device ID
            // [Track number: E#30810]
            ".*OVClassLoadNetworkTest.*LoadNetworkHETEROWithBigDeviceIDThrows.*",

            // TODO VPUX Plugin doesn't handle DEVICE_ID in QueryNetwork implementation
            // [Track number: E#30815]
            ".*OVClassQueryNetworkTest.*",

            // Cannot detect vpu platform when it's not passed
            // Skip tests on Yocto which passes device without platform
            // [Track number: E#12774]
            ".*IEClassLoadNetworkTest.LoadNetworkWithDeviceIDNoThrow.*",
            ".*IEClassLoadNetworkTest.LoadNetworkWithBigDeviceIDThrows.*",
            ".*IEClassLoadNetworkTest.LoadNetworkWithInvalidDeviceIDThrows.*",

            // [Track number: E#28335]
            ".*smoke_LoadNetworkToDefaultDeviceNoThrow.*",

            // [Track number: E#32241]
            ".*LoadNetwork.*CheckDeviceInBlob.*",

            // double free detected
            // [Track number: S#27343]
            ".*InferConfigInTests\\.CanInferWithConfig.*",
            ".*InferConfigTests\\.withoutExclusiveAsyncRequests.*",
            ".*InferConfigTests\\.canSetExclusiveAsyncRequests.*",

            // TODO: GetExecGraphInfo function is not implemented for VPUX plugin
            ".*checkGetExecGraphInfoIsNotNullptr.*",
            ".*CanCreateTwoExeNetworksAndCheckFunction.*",
            ".*CheckExecGraphInfo.*",
            ".*canLoadCorrectNetworkToGetExecutable.*",

            // [Track number: E#31074]
            ".*checkInferTime.*",
            ".*OVExecGraphImportExportTest.*",

            // TODO: GetMetric function is not fully implemented for ExecutableNetwork interface (implemented only for vpux plugin)
            ".*ExecutableNetworkBaseTest.checkGetMetric.*",
            ".*OVHoldersTest.*LoadedAny.*",
            ".*OVClassExecutableNetworkGetMetricTest.*",

            // TODO: SetConfig function is not implemented for ExecutableNetwork interface (implemented only for vpux plugin)
            ".*ExecutableNetworkBaseTest.canSetConfigToExecNet.*",
            ".*ExecutableNetworkBaseTest.canSetConfigToExecNetAndCheckConfigAndCheck.*",
            ".*CanSetConfigToExecNet.*",
            ".*OVClassExecutableNetworkGetMetricTest.*",
            ".*OVClassExecutableNetworkGetConfigTest.*",

            // TODO Exception "Not implemented"
            // [Track number: E#30822]
            ".*OVClassNetworkTestP.*LoadNetworkCreateDefaultExecGraphResult.*",

            // Async tests failed on dKMB
            // TODO: [Track number: S#14836]
            ".*ExclusiveAsyncRequests.*",
            ".*MultithreadingTests.*",

            // This is openvino specific test
            ".*ExecutableNetworkBaseTest.canExport.*",

            // TODO: Issue: 63469
            ".*KmbConversionLayerTest.*ConvertLike.*",

            // TensorIterator layer is not supported
            ".*ReturnResultNotReadyFromWaitInAsyncModeForTooSmallTimeout.*",
            ".*OVInferRequestDynamicTests.*",
            ".*OVInferenceChaining.*",

            // TODO: EISW-25458
            ".*OVInferRequestPerfCountersTest.*",

            // TODO: EISW-29542
            ".*smoke_Auto_BehaviorTests.*",

            // Current OV logic with FULL_DEVICE_NAME metric differs from VPUX Plugin
            // Plugin throws an exception in case of absence VPUX devices in system
            ".*IEClassGetMetricTest_FULL_DEVICE_NAME.*GetMetricAndPrintNoThrow.*",

            // Tests with unsupported precision
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

            // TODO Exception during loading to the device
            // [Track number: E#32075]
            ".*OVClassLoadNetworkTest.*LoadNetworkHETEROwithMULTINoThrow.*",
            ".*OVClassLoadNetworkTest.*LoadNetworkMULTIwithHETERONoThrow.*",
            }
        );

        //
        // Conditionally disabled test patterns
        //

        _skipRegistry.addPatterns(
            backendName.isEmpty(),  
            "backend is empty (no device)",
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

                // Exception in case of network compilation without devices in system
                // [Track number: E#30824]
                ".*OVClassImportExportTestP.*",
                ".*OVClassLoadNetworkTest.*LoadNetwork.*",
            }
        );

        _skipRegistry.addPatterns(
            backendName.isZero(),  
            "CumSum layer is not supported by MTL platform",
            {
                ".*SetBlobTest.*",
            }
        );

        _skipRegistry.addPatterns(
            backendName.isZero(),  
            "TensorIterator layer is not supported by MTL/dKMB platform",
            {
                ".*SetBlobTest.*",
            }
        );

        _skipRegistry.addPatterns(
            backendName.isZero(),
            "Abs layer is not supported by MTL/dKMB platform",
            {
                ".*PrePostProcessTest.*"
            }
        );

        _skipRegistry.addPatterns(
                backendName.isZero(),
                "Convert layer is not supported by MTL/dKMB platform",
                {
                        ".*PreprocessingPrecisionConvertTest.*",
                        ".*InferRequestPreprocess.*"
                }
        );

        // [Track number: E#32244]
        _skipRegistry.addPatterns(
                backendName.isZero(),
                "Device failure with 0x700000001",
                {
                        "smoke_BehaviorTests.InferRequestPerfCountersTest.*",
                }
        );

        _skipRegistry.addPatterns(
            platform.isARM(),  
            "CumSum layer is not supported by ARM platform",
            {
                ".*SetBlobTest.CompareWithRefs.*",
            }
        );

        // TODO: [Track number: E#26428]
        _skipRegistry.addPatterns(
            platform.isARM(),
            "LoadNetwork throws an exception",
            {
                ".*KmbGatherLayerTest.CompareWithRefs/.*",
            }
        );

        _skipRegistry.addPatterns(
            backendName.getAvailableDevices().size() > 1,
            "Some VPUX Plugin metrics require single device to work in auto mode or set particular device",
            {
            ".*OVClassGetConfigTest.*GetConfigNoThrow.*",
            ".*OVClassGetConfigTest.*GetConfigHeteroNoThrow.*",
            }
        );

        return _skipRegistry;
    }();

    std::vector<std::string> matchingPatterns;
    const auto currentTestName = getCurrentTestName();
    matchingPatterns.emplace_back(skipRegistry.getMatchingPattern(currentTestName));

    return matchingPatterns;
}
