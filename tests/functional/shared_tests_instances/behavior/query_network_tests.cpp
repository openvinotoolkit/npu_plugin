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

#include <vector>

#include "hetero/query_network.hpp"
#include "ngraph_functions/builders.hpp"
#include <ngraph_functions/subgraph_builders.hpp>
#include "vpux_private_config.hpp"

namespace {
using namespace HeteroTests;

auto ConvBias = ngraph::builder::subgraph::makeConvBias();

TEST_P(QueryNetworkTest, queryNetworkResultContainAllAndOnlyInputLayers_MLIR) {
    // Skip test according to plugin specific disabledTestPatterns() (if any)
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    auto& param = GetParam();
    std::map<std::string, std::string> config;
    config[VPUX_CONFIG_KEY(COMPILER_TYPE)] = VPUX_CONFIG_VALUE(MLIR);
    auto queryNetworkResult = PluginCache::get().ie()->QueryNetwork(cnnNetwork, std::get<Plugin>(param), config);
    ASSERT_NE(nullptr, cnnNetwork.getFunction());
    std::set<std::string> expectedLayers;
    for (auto&& node : function->get_ops()) {
        expectedLayers.insert(node->get_friendly_name());
    }
    std::set<std::string> actualLayers;
    for (auto&& res : queryNetworkResult.supportedLayersMap) {
        actualLayers.insert(res.first);
    }
    ASSERT_EQ(expectedLayers, actualLayers);
}

INSTANTIATE_TEST_CASE_P(smoke_BEHTests, QueryNetworkTest,
                        ::testing::Combine(
                                ::testing::Values("VPUX", "HETERO:VPUX", "MULTI:VPUX"),
                                ::testing::Values(ConvBias)),
                        QueryNetworkTest::getTestCaseName);
} // namespace
