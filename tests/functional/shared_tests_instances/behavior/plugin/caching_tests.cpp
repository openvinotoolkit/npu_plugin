// Copyright (C) 2018-2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "behavior/plugin/caching_tests.hpp"
#include <algorithm>
#include <vector>
#include "common/utils.hpp"

using namespace LayerTestsDefinitions;

namespace {
static const std::vector<ngraph::element::Type> nightly_precisionsKeemBay = {
        ngraph::element::f32,
        ngraph::element::f16,
};

static const std::vector<ngraph::element::Type> smoke_precisionsKeemBay = {
        ngraph::element::f32,
};

static const std::vector<std::size_t> batchSizesKeemBay = {1};

static std::vector<nGraphFunctionWithName> smoke_functions() {
    auto funcs = LoadNetworkCacheTestBase::getStandardFunctions();
    if (funcs.size() > 1) {
        funcs.erase(funcs.begin() + 1, funcs.end());
    }
    return funcs;
}

static std::vector<nGraphFunctionWithName> keembay_functions() {
    auto funcs = LoadNetworkCacheTestBase::getStandardFunctions();

    std::vector<nGraphFunctionWithName>::iterator it =
            remove_if(funcs.begin(), funcs.end(), [](nGraphFunctionWithName func) {
                std::vector<std::string> bad_layers{"SimpleFunctionRelu", "TIwithLSTMcell1", "2InputSubtract",
                                                    "ReadConcatSplitAssign", "MatMulBias"};
                return std::find(bad_layers.begin(), bad_layers.end(), std::get<1>(func)) != bad_layers.end();
            });

    funcs.erase(it, funcs.end());

    return funcs;
}

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_KeemBay, LoadNetworkCacheTestBase,
                         ::testing::Combine(::testing::ValuesIn(smoke_functions()),
                                            ::testing::ValuesIn(smoke_precisionsKeemBay),
                                            ::testing::ValuesIn(batchSizesKeemBay),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU)),
                         LoadNetworkCacheTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_CachingSupportCase_KeemBay, LoadNetworkCacheTestBase,
                         ::testing::Combine(::testing::ValuesIn(keembay_functions()),
                                            ::testing::ValuesIn(nightly_precisionsKeemBay),
                                            ::testing::ValuesIn(batchSizesKeemBay),
                                            ::testing::Values(ov::test::utils::DEVICE_NPU)),
                         LoadNetworkCacheTestBase::getTestCaseName);
}  // namespace
