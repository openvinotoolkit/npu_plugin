// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/skip_tests_config.hpp"

#include <string>
#include <vector>

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
        }
    }

    auto getName() const {
        return _name;
    }

    bool isEmpty() const noexcept {
        return _name.empty();
    }

    bool isZero() const noexcept {
        return _name == "LEVEL0";
    }

    bool isVpual() const noexcept {
        return _name == "VPUAL";
    }

    bool isHddl2() const noexcept {
        return _name == "HDDL2";
    }

    bool isEmulator() const noexcept {
        return _name == "EMULATOR";
    }

private:
    std::string _name;
};

class TestsSkipper{
public:
    TestsSkipper(std::vector<std::string>& registry) : _registry{registry} {
    }

    void addPatterns(std::vector<std::string>&& patternsToSkip) const {
        _registry.insert(
            _registry.end(),
            std::make_move_iterator(patternsToSkip.begin()),
            std::make_move_iterator(patternsToSkip.end())
        );
    }
    
    void addPatterns(bool conditionFlag,
                     const std::string& comment,
                     std::vector<std::string>&& patternsToSkip) const {
        if (conditionFlag) {
            std::cout << comment << "\n";
            addPatterns(std::move(patternsToSkip));
        }
    }

private:
    std::vector<std::string>& _registry;
};

std::vector<std::string> disabledTestPatterns() {
    static const auto allDisabledTestPatterns = []() {
        std::vector<std::string> disabledTests;
        TestsSkipper skipper(disabledTests);
        const BackendName backendName;
        
        //
        //  Disabled test patterns
        //

        skipper.addPatterns({
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

        skipper.addPatterns(
            backendName.isEmpty(),  
            "backend is empty (no device)",
            {
                // Cannot run InferRequest tests without a device to infer to
                ".*InferRequest.*",
                ".*ExecutableNetworkBaseTest.*",
                ".*ExecNetSetPrecision.*",
                ".*VpuxInferRequestCallbackTests.*"
            }
        );

        skipper.addPatterns(
            backendName.isZero(),  
            "* CumSum layer is not supported by MTL platform *",
            {
                ".*VpuxBehaviorTestsSetBlob.*",
            }
        );

        return disabledTests;
    }( );

    return allDisabledTestPatterns;
}
