//
// Copyright 2022 Intel Corporation.
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

#include "behavior/ov_infer_request/infer_request_dynamic.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> configs = {
    {}
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferRequestDynamicTests,
                        ::testing::Combine(
                                ::testing::Values(ngraph::builder::subgraph::makeSplitConvConcat()),
                                ::testing::Values(std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>>{{{1, 4, 20, 20}, {1, 10, 18, 18}},
                                                                                                                   {{2, 4, 20, 20}, {2, 10, 18, 18}}}),
                                ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                ::testing::ValuesIn(configs)),
                        OVInferRequestDynamicTests::getTestCaseName);

}  // namespace
