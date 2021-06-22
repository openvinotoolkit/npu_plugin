// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/caching_tests.hpp"

using namespace LayerTestsDefinitions;

namespace {
static const std::vector<ngraph::element::Type> netPrecisions = {
        ngraph::element::f32,
};

static const std::vector<std::size_t> netBatchSizes = {
        1,
};

static std::vector<nGraphFunctionWithName> smoke_functions() {
    auto funcs = LoadNetworkCacheTestBase::getStandardFunctions();
    if (funcs.size() > 1) {
        funcs.erase(funcs.begin() + 1, funcs.end());
    }
    return funcs;
}

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_KMB, LoadNetworkCacheTestBase,
                        ::testing::Combine(
                                ::testing::ValuesIn(smoke_functions()),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(netBatchSizes),
                                ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY)),
                        LoadNetworkCacheTestBase::getTestCaseName);
} // namespace
