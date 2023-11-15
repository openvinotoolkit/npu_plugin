//
// Copyright (C) Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <array>
#include <thread>

#include "base/ov_behavior_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "ngraph_functions/builders.hpp"
#include "vpu_test_env_cfg.hpp"

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <exception>

#include <openvino/core/any.hpp>
#include <openvino/core/node_vector.hpp>
#include <openvino/op/op.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/core.hpp>

using CompilationParams = std::tuple<std::string,  // Device name
                                     ov::AnyMap    // Config
                                     >;

using ::testing::AllOf;
using ::testing::HasSubstr;

namespace ov {
namespace test {
namespace behavior {
class InferRequestRunTests :
        public ov::test::behavior::OVPluginTestBase,
        public testing::WithParamInterface<CompilationParams> {
protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    ov::AnyMap configuration;
    std::shared_ptr<ov::Model> function;
    ov::CompiledModel compiledModel;
    ov::Output<const ov::Node> input;
    ov::Output<const ov::Node> output;

public:
    static std::string getTestCaseName(testing::TestParamInfo<CompilationParams> obj) {
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '_');

        std::ostringstream result;
        result << "targetDevice=" << LayerTestsUtils::getDeviceNameTestCase(targetDevice) << "_";
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
            }
        }

        return result.str();
    }

    void SetUp() override {
        std::tie(target_device, configuration) = this->GetParam();

        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        OVPluginTestBase::SetUp();
        function = getDefaultNGraphFunctionForTheDevice(target_device);
        ov::AnyMap params;
        for (auto&& v : configuration) {
            params.emplace(v.first, v.second);
        }
    }

    void TearDown() override {
        if (!configuration.empty()) {
            utils::PluginCache::get().reset();
        }
        APIBaseTest::TearDown();
    }
};

TEST_P(InferRequestRunTests, AllocatorCanDisposeBlobWhenOnlyInferRequestIsInScope) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED() {
        ov::InferRequest req;
        ov::Tensor outputTensor;
        {
            core.reset();
            PluginCache::get().reset();
        }
    }
    std::cout << "Plugin should be unloaded from memory at this point" << std::endl;
}

TEST_P(InferRequestRunTests, MultipleExecutorStreamsTestsSyncInfers) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Load CNNNetwork to target plugins
    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(function, target_device, configuration));
    OV_ASSERT_NO_THROW(input = compiledModel.input());
    OV_ASSERT_NO_THROW(output = compiledModel.output());

    // Create InferRequests
    const int inferReqNumber = 256;
    std::array<ov::InferRequest, inferReqNumber> inferReqs;
    std::array<std::thread, inferReqNumber> inferReqsThreads;
    for (int i = 0; i < inferReqNumber; ++i) {
        OV_ASSERT_NO_THROW(inferReqs[i] = compiledModel.create_infer_request());
        OV_ASSERT_NO_THROW(inferReqs[i].get_tensor(input));
    }

    for (int i = 0; i < inferReqNumber; ++i) {
        ov::InferRequest& infReq = inferReqs[i];
        inferReqsThreads[i] = std::thread([&infReq]() -> void {
            OV_ASSERT_NO_THROW(infReq.infer());
        });
    }

    for (int i = 0; i < inferReqNumber; ++i) {
        inferReqsThreads[i].join();
        OV_ASSERT_NO_THROW(inferReqs[i].get_tensor(output));
    }
}

TEST_P(InferRequestRunTests, MultipleExecutorStreamsTestsAsyncInfers) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Load CNNNetwork to target plugins
    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(function, target_device, configuration));
    OV_ASSERT_NO_THROW(input = compiledModel.input());
    OV_ASSERT_NO_THROW(output = compiledModel.output());

    // Create InferRequests
    const int inferReqNumber = 256;
    std::array<ov::InferRequest, inferReqNumber> inferReqs;
    for (int i = 0; i < inferReqNumber; ++i) {
        OV_ASSERT_NO_THROW(inferReqs[i] = compiledModel.create_infer_request());
        OV_ASSERT_NO_THROW(inferReqs[i].get_tensor(input));
    }

    for (int i = 0; i < inferReqNumber; ++i) {
        OV_ASSERT_NO_THROW(inferReqs[i].start_async());
    }

    for (int i = 0; i < inferReqNumber; ++i) {
        inferReqs[i].wait();
        OV_ASSERT_NO_THROW(inferReqs[i].get_tensor(output));
    }
}

TEST_P(InferRequestRunTests, MultipleExecutorTestsSyncInfers) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Load CNNNetwork to target plugins
    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(function, target_device, configuration));
    OV_ASSERT_NO_THROW(input = compiledModel.input());
    OV_ASSERT_NO_THROW(output = compiledModel.output());

    // Create InferRequests
    const int inferReqNumber = 256;
    ov::InferRequest inferReq;
    ov::Tensor input_tensor;
    for (int i = 0; i < inferReqNumber; ++i) {
        OV_ASSERT_NO_THROW(inferReq = compiledModel.create_infer_request());
        OV_ASSERT_NO_THROW(input_tensor = inferReq.get_tensor(input));
        OV_ASSERT_NO_THROW(inferReq.set_input_tensor(input_tensor));
        OV_ASSERT_NO_THROW(inferReq.infer());
        OV_ASSERT_NO_THROW(inferReq.get_tensor(output));
    }
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
