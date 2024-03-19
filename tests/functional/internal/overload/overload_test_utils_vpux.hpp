// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "base/behavior_test_utils.hpp"
#include "base/ov_behavior_test_utils.hpp"
#include "ov_models/subgraph_builders.hpp"

namespace ov {
namespace test {
namespace behavior {

inline std::shared_ptr<ngraph::Function> getDefaultNGraphFunctionForTheDeviceVpux(
        std::vector<size_t> inputShape = {1, 2, 32, 32}, ngraph::element::Type_t ngPrc = ngraph::element::Type_t::f32) {
    return ngraph::builder::subgraph::makeConvPoolRelu(inputShape, ngPrc);
}

class OVInferRequestTestsVpux : public OVInferRequestTests {
public:
    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDeviceVpux();
        ov::AnyMap params;
        for (auto&& v : configuration) {
            params.emplace(v.first, v.second);
        }
        execNet = core->compile_model(function, target_device, params);
    }
};

}  // namespace behavior
}  // namespace test
}  // namespace ov

namespace BehaviorTestsUtils {

class InferRequestTestsVpux : public InferRequestTests {
public:
    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        function = ov::test::behavior::getDefaultNGraphFunctionForTheDeviceVpux();
        cnnNet = InferenceEngine::CNNNetwork(function);
        // Load CNNNetwork to target plugins
        execNet = ie->LoadNetwork(cnnNet, target_device, configuration);
    }
};

}  // namespace BehaviorTestsUtils