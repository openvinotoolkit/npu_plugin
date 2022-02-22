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

#include <base/behavior_test_utils.hpp>

#include "behavior/plugin/set_preprocess.hpp"
#include "common/functions.h"

using namespace LayerTestsUtils;

using namespace BehaviorTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {{}};

const std::vector<InferenceEngine::Precision> ioPrecisions = {
        InferenceEngine::Precision::U8
};

const std::vector<InferenceEngine::Layout> netLayouts = {
        InferenceEngine::Layout::NCHW
};

const std::vector<InferenceEngine::Layout> ioLayouts = {
        InferenceEngine::Layout::NCHW,
        InferenceEngine::Layout::NHWC
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPreprocessConversionTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::ValuesIn(ioPrecisions),
                                 ::testing::ValuesIn(ioPrecisions),
                                 ::testing::ValuesIn(netLayouts),
                                 ::testing::ValuesIn(ioLayouts),
                                 ::testing::ValuesIn(ioLayouts),
                                 ::testing::Bool(),
                                 ::testing::Bool(),
                                 ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                 ::testing::ValuesIn(configs)),
                         InferRequestPreprocessConversionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, InferRequestPreprocessDynamicallyInSetBlobTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Bool(),
                                 ::testing::Bool(),
                                 ::testing::ValuesIn(netLayouts),
                                 ::testing::Values(false),
                                 ::testing::Values(false),
                                 ::testing::Values(true), // only SetBlob
                                 ::testing::Values(true), // only SetBlob
                                 ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                 ::testing::ValuesIn(configs)),
                         InferRequestPreprocessDynamicallyInSetBlobTest::getTestCaseName);

}  // namespace
