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

namespace {
using namespace HeteroTests;

auto ConvBias = ngraph::builder::subgraph::makeConvBias();

INSTANTIATE_TEST_CASE_P(smoke_BEHTests, QueryNetworkTest,
                        ::testing::Combine(
                                ::testing::Values("VPUX", "HETERO:VPUX", "MULTI:VPUX"),
                                ::testing::Values(ConvBias)),
                        QueryNetworkTest::getTestCaseName);
} // namespace
