// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/memory_LSTMCell.hpp"
#include <subgraph_tests/memory_LSTMCell.hpp>
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

std::vector<ngraph::helpers::MemoryTransformation> transformation{
        ngraph::helpers::MemoryTransformation::NONE,
};

std::vector<size_t> input_sizes = {80, 32, 64, 100, 25};

std::vector<size_t> hidden_sizes = {
        128, 200, 300, 24, 32,
};

std::map<std::string, std::string> additional_config = {};

// Tests fail with threshold 0.01 and fail to infer result type on VPUX37XX.
// E#62473
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_MemoryLSTMCellTest, MemoryLSTMCellTest,
                         ::testing::Combine(::testing::ValuesIn(transformation),
                                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::ValuesIn(input_sizes), ::testing::ValuesIn(hidden_sizes),
                                            ::testing::Values(additional_config)),
                         MemoryLSTMCellTest::getTestCaseName);
