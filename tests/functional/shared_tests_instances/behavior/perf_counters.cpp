// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/perf_counters.hpp"
#include "kmb_layer_test.hpp"

using namespace LayerTestsUtils;

namespace BehaviorTestsDefinitions {

class VpuxPerfCountersTest :
        public PerfCountersTest,
        virtual public KmbLayerTestsCommon {

        void TearDown() override 
        {
            KmbLayerTestsCommon::TearDown();     
        }
};

TEST_P(VpuxPerfCountersTest, NotEmptyWhenExecuted) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    // Set up config
    KmbLayerTestsCommon::configuration.insert(PerfCountersTest::configuration.begin(),
                                              PerfCountersTest::configuration.end());   
    KmbLayerTestsCommon::configuration.insert({ InferenceEngine::PluginConfigParams::KEY_PERF_COUNT,
                                                InferenceEngine::PluginConfigParams::YES });
    KmbLayerTestsCommon::targetDevice = LayerTestsUtils::testPlatformTargetDevice;
    // Set up ngrpah::Function
    KmbLayerTestsCommon::function = ngraph::builder::subgraph::makeConvPoolRelu();
    KmbLayerTestsCommon::Run(); 
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap;
    perfMap = KmbLayerTestsCommon::inferRequest.GetPerformanceCounts();
    ASSERT_NE(perfMap.size(), 0);
}

}// namespace BehaviorTestsDefinitions

using namespace BehaviorTestsDefinitions;
const LayerTestsUtils::KmbTestEnvConfig envConfig;

namespace {
    const std::vector<std::map<std::string, std::string>> configs = {
            {}
    };

    INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, VpuxPerfCountersTest,
                            ::testing::Combine(
                                    ::testing::Values(InferenceEngine::Precision::FP32),
                                    ::testing::Values(LayerTestsUtils::testPlatformTargetDevice),
                                    ::testing::ValuesIn(configs)),
                            VpuxPerfCountersTest::getTestCaseName);

}  // namespace

