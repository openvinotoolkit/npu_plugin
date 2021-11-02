// Copyright (C) 2021 Intel Corporation
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
        } else {
            std::cout << "Failed to get IE Core!" << std::endl;
        }
    }

    auto getName() const {
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
            for (const auto& pattern : entry.patterns) {
                std::regex re(pattern);
                if (std::regex_match(testName, re)) {
                    std::cout << entry.comment << std::endl;
                    return pattern;
                }
            }
        }

        return std::string{};
    }

private:
    struct Entry {
        Entry(std::string&& comment, std::vector<std::string>&& patterns) :
              comment{std::move(comment)}
            , patterns{std::move(patterns)} { }
            
        std::string comment;
        std::vector<std::string> patterns;
    };

    std::vector<Entry> _registry;
};

std::vector<std::string> disabledTestPatterns() {
    // Initialize skip registry
    static const auto skipRegistry = []() {
        SkipRegistry _skipRegistry;
        const BackendName backendName;
        
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

            // TODO: Issue: 63469
            ".*KmbConversionLayerTest.*ConvertLike.*"
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
                ".*ExecutableNetworkBaseTest.*",
                ".*ExecNetSetPrecision.*",
                ".*VpuxInferRequestCallbackTests.*",
                ".*VpuxInferRequestConfigTest.*"
            }
        );

        _skipRegistry.addPatterns(
            backendName.isZero(),  
            "CumSum layer is not supported by MTL platform",
            {
                ".*VpuxBehaviorTestsSetBlob.*",
            }
        );

        return _skipRegistry;
    }( );

    const auto* currentTestInfo = ::testing::UnitTest::GetInstance()->current_test_info();
    const auto currentTestName = currentTestInfo->test_case_name()
                          + std::string(".") + currentTestInfo->name();

    std::vector<std::string> matching_patterns;
    matching_patterns.emplace_back(skipRegistry.getMatchingPattern(currentTestName));

    return matching_patterns;
}
