//
// Copyright (C) 2018-2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include "shared_test_classes/subgraph/memory_LSTMCell.hpp"
#include <subgraph_tests/memory_LSTMCell.hpp>
#include "common_test_utils/test_constants.hpp"
#include "vpu_test_env_cfg.hpp"

using namespace SubgraphTestsDefinitions;

static std::string getTestCaseName(testing::TestParamInfo<memoryLSTMCellParams> obj) {
    ngraph::helpers::MemoryTransformation memoryTransform;
    std::string targetDevice;
    InferenceEngine::Precision precision;
    size_t inputSize;
    size_t outputSize;
    std::map<std::string, std::string> configuration;
    std::tie(memoryTransform, targetDevice, precision, inputSize, outputSize, configuration) = obj.param;
    std::ostringstream result;
    result << "targetDevice=" << LayerTestsUtils::getDeviceNameTestCase(targetDevice) << "_";
    result << "inSize=" << inputSize << "_";
    result << "outSize=" << outputSize << "_";
    return result.str();
}

std::vector<ngraph::helpers::MemoryTransformation> transformation{
        ngraph::helpers::MemoryTransformation::NONE,
};

std::vector<size_t> input_sizes = {80, 32, 64, 100, 25};

std::vector<size_t> hidden_sizes = {
        128, 200, 300, 24, 32,
};

std::map<std::string, std::string> additional_config = {};

INSTANTIATE_TEST_SUITE_P(smoke_MemoryLSTMCellTest, MemoryLSTMCellTest,
                         ::testing::Combine(::testing::ValuesIn(transformation),
                                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::ValuesIn(input_sizes), ::testing::ValuesIn(hidden_sizes),
                                            ::testing::Values(additional_config)),
                         getTestCaseName);
