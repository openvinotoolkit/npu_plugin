//
// Copyright 2021 Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
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

private:
    std::string _name;
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
    
    std::cout << "NAME:" << currentTestName << std::endl;
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

            // Cannot detect vpu platform when it's not passed
            // Skip tests on Yocto which passes device without platform
            // [Track number: E#12774]
            ".*IEClassLoadNetworkTest.LoadNetworkWithDeviceIDNoThrow.*",
            ".*IEClassLoadNetworkTest.LoadNetworkWithBigDeviceIDThrows.*",
            ".*IEClassLoadNetworkTest.LoadNetworkWithInvalidDeviceIDThrows.*",

            // double free detected
            // [Track number: S#27343]
            ".*InferConfigInTests\\.CanInferWithConfig.*",
            ".*InferConfigTests\\.withoutExclusiveAsyncRequests.*",
            ".*InferConfigTests\\.canSetExclusiveAsyncRequests.*",

            // TODO Add safe Softplus support
            ".*ActivationLayerTest.*SoftPlus.*",

            // TODO: GetExecGraphInfo function is not implemented for VPUX plugin
            ".*checkGetExecGraphInfoIsNotNullptr.*",
            ".*CanCreateTwoExeNetworksAndCheckFunction.*",
            ".*CheckExecGraphInfo.*",
            ".*canLoadCorrectNetworkToGetExecutable.*",

            // TODO: GetMetric function is not fully implemented for ExecutableNetwork interface (implemented only for vpux plugin)
            ".*ExecutableNetworkBaseTest.checkGetMetric.*",

            // TODO: SetConfig function is not implemented for ExecutableNetwork interface (implemented only for vpux plugin)
            ".*ExecutableNetworkBaseTest.canSetConfigToExecNet.*",
            ".*ExecutableNetworkBaseTest.canSetConfigToExecNetAndCheckConfigAndCheck.*",

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
            ".*SetBlobTest.*",
            ".*OVInferRequestDynamicTests.*",
            ".*OVInferenceChaining.*",
            ".*OVInferRequestCallbackTests.ReturnResultNotReadyFromWaitInAsyncModeForTooSmallTimeout.*",
            
            // CumSum layer is not supported
            ".*SetBlobTest.*",

            // Abs layer is not supported by MTL/dKMB platform
            ".*PrePostProcessTest.*",

            // TODO: Fix failing OVInferRequestPerfCountersTest with multi device
            ".*smoke_Multi_BehaviorTests/OVInferRequestPerfCountersTest.*"
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
                ".*ExecutableNetworkBaseTest.*",
                ".*ExecNetSetPrecision.*",
                ".*SetBlobTest.*",
                ".*InferRequestCallbackTests.*",
                ".*PrePostProcessTest.*"
            }
        );

        return _skipRegistry;
    }();

    std::vector<std::string> matchingPatterns;
    const auto currentTestName = getCurrentTestName();
    matchingPatterns.emplace_back(skipRegistry.getMatchingPattern(currentTestName));

    return matchingPatterns;
}
