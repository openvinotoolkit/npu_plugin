// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include "behavior/ov_plugin/life_time.hpp"

namespace ov {
namespace test {
namespace behavior {

class OVHoldersTestVpux : public OVHoldersTest {
public:
    void SetUp() override {
        target_device = this->GetParam();
        APIBaseTest::SetUp();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        deathTestStyle = ::testing::GTEST_FLAG(death_test_style);
        if (deathTestStyle == "fast") {
            ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
        }
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }
};

#define EXPECT_NO_CRASH(_statement) EXPECT_EXIT(_statement; exit(0), testing::ExitedWithCode(0), "")

static void release_order_test(std::vector<std::size_t> order, const std::string& deviceName,
                               std::shared_ptr<ngraph::Function> function) {
    ov::AnyVector objects;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, deviceName);
        auto request = compiled_model.create_infer_request();

        objects = {core, compiled_model, request};
    }
    for (auto&& i : order) {
        objects.at(i) = {};
    }
}

#ifndef __EMSCRIPTEN__

TEST_P(OVHoldersTestVpux, Orders) {
    std::vector<std::string> objects{"core", "compiled_model", "request"};
    std::vector<std::size_t> order(objects.size());
    std::iota(order.begin(), order.end(), 0);
    do {
        std::stringstream order_str;
        for (auto&& i : order) {
            order_str << objects.at(i) << " ";
        }
        EXPECT_NO_CRASH(release_order_test(order, target_device, function)) << "for order: " << order_str.str();
    } while (std::next_permutation(order.begin(), order.end()));
}

#endif  // __EMSCRIPTEN__

TEST_P(OVHoldersTestVpux, LoadedState) {
    std::vector<ov::VariableState> states;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, target_device);
        auto request = compiled_model.create_infer_request();
        try {
            states = request.query_state();
        } catch (...) {
        }
    }
}

TEST_P(OVHoldersTestVpux, LoadedTensor) {
    ov::Tensor tensor;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, target_device);
        auto request = compiled_model.create_infer_request();
        tensor = request.get_input_tensor();
    }
}

TEST_P(OVHoldersTestVpux, LoadedAny) {
    ov::Any any;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, target_device);
        any = compiled_model.get_property(ov::supported_properties.name());
    }
}

TEST_P(OVHoldersTestVpux, LoadedRemoteContext) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    ov::RemoteContext ctx;
    {
        ov::Core core = createCoreWithTemplate();
        auto compiled_model = core.compile_model(function, target_device);
        try {
            ctx = compiled_model.get_context();
        } catch (...) {
        }
    }
}

class OVHoldersTestOnImportedNetworkVpux : public OVHoldersTestOnImportedNetwork {
public:
    void SetUp() override {
        target_device = this->GetParam();
        APIBaseTest::SetUp();
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        deathTestStyle = ::testing::GTEST_FLAG(death_test_style);
        if (deathTestStyle == "fast") {
            ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
        }
        function = ngraph::builder::subgraph::makeConvPoolRelu();
    }
};

TEST_P(OVHoldersTestOnImportedNetworkVpux, LoadedTensor) {
    ov::Core core = createCoreWithTemplate();
    std::stringstream stream;
    {
        auto compiled_model = core.compile_model(function, target_device);
        compiled_model.export_model(stream);
    }
    auto compiled_model = core.import_model(stream, target_device);
    auto request = compiled_model.create_infer_request();
    ov::Tensor tensor = request.get_input_tensor();
}

TEST_P(OVHoldersTestOnImportedNetworkVpux, CreateRequestWithCoreRemoved) {
    ov::Core core = createCoreWithTemplate();
    std::stringstream stream;
    {
        auto compiled_model = core.compile_model(function, target_device);
        compiled_model.export_model(stream);
    }
    auto compiled_model = core.import_model(stream, target_device);
    core = ov::Core{};
    auto request = compiled_model.create_infer_request();
}

}  // namespace behavior
}  // namespace test
}  // namespace ov
