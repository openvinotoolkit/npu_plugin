// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/perf_counters.hpp"
#include "kmb_layer_test.hpp"
#include "common/functions.h"

using namespace LayerTestsUtils;

namespace BehaviorTestsDefinitions {

class VpuxPerfCountersTest :
        public PerfCountersTest, 
        virtual public KmbLayerTestsCommon {
                
    void SetUp() override {
        std::tie(std::ignore, // Network precision
                 KmbLayerTestsCommon::targetDevice,
                 KmbLayerTestsCommon::configuration) = this->GetParam();

        KmbLayerTestsCommon::configuration.insert({InferenceEngine::PluginConfigParams::KEY_PERF_COUNT,
                                                   InferenceEngine::PluginConfigParams::YES});

        KmbLayerTestsCommon::function = ngraph::builder::subgraph::makeConvPoolRelu();
    }

    void SkipBeforeLoad() override {
        const std::string backendName = getBackendName(*core);
        const auto noDevice = backendName.empty();
        if (noDevice) {
            throw LayerTestsUtils::KmbSkipTestException("backend is empty (no device)");
        }
    }

    void TearDown() override {
        KmbLayerTestsCommon::TearDown();
    }
};

TEST_P(VpuxPerfCountersTest, NotEmptyWhenExecuted) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    KmbLayerTestsCommon::Run();
    if (KmbLayerTestsCommon::inferRequest) {
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
        perfMap = KmbLayerTestsCommon::inferRequest.GetPerformanceCounts();
        ASSERT_NE(perfMap.size(), 0);
    }
}

}  // namespace BehaviorTestsDefinitions

using namespace BehaviorTestsDefinitions;

namespace {
const std::vector<std::map<std::string, std::string>> configs = {{}};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, VpuxPerfCountersTest,
                        ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP32),
                                           ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                           ::testing::ValuesIn(configs)),
                        VpuxPerfCountersTest::getTestCaseName);

}  // namespace
