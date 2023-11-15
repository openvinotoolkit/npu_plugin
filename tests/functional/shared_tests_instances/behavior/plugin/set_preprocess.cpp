//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <base/behavior_test_utils.hpp>

#include "behavior/plugin/set_preprocess.hpp"
#include "common/functions.h"

using namespace LayerTestsUtils;

using namespace BehaviorTestsDefinitions;

namespace {
// Precision types
/*
    {InferenceEngine::Precision::I8,   InferenceEngine::Precision::U8,
    InferenceEngine::Precision::I16,  InferenceEngine::Precision::U16,
    InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32};
*/
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16,
                                                               InferenceEngine::Precision::FP32};

/* The Input and Output precisions conversion tests only cover U8 & FP32 currently
 Any other value will lead to asserts getting triggered */
const std::vector<InferenceEngine::Precision> inputPrecisions = {InferenceEngine::Precision::U8,
                                                                 InferenceEngine::Precision::FP32};
const std::vector<InferenceEngine::Precision> outputPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<std::map<std::string, std::string>> configs = {{}};

const std::vector<std::map<std::string, std::string>> autoConfigs = {
        {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES, CommonTestUtils::DEVICE_KEEMBAY},
         {InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
          CommonTestUtils::DEVICE_KEEMBAY + std::string(",") + CommonTestUtils::DEVICE_CPU}}};

// Layout types:
/*
    {InferenceEngine::Layout::C,
    InferenceEngine::Layout::NC, InferenceEngine::Layout::CN, InferenceEngine::Layout::HW,
    InferenceEngine::Layout::CHW, InferenceEngine::Layout::HWC,
    InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC,
    InferenceEngine::Layout::NCDHW, InferenceEngine::Layout::NDHWC};
*/
const std::vector<InferenceEngine::Layout> netLayouts = {
        InferenceEngine::Layout::C,    InferenceEngine::Layout::NC,   InferenceEngine::Layout::CN,
        InferenceEngine::Layout::HW,   InferenceEngine::Layout::CHW,  InferenceEngine::Layout::HWC,
        InferenceEngine::Layout::NCHW, InferenceEngine::Layout::NHWC, InferenceEngine::Layout::NCDHW,
        InferenceEngine::Layout::NDHWC};

/*
The input layouts can only be set to NCHW or NHWC, due to the following
way in which the param is created to have 4 dimensions for this IR:
    unsigned int shape_size = 9, channels = 3, batch = 1, offset = 0;
    ngraph::PartialShape shape({batch, channels, shape_size, shape_size});
    auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
*/
const std::vector<InferenceEngine::Layout> inputLayouts = {InferenceEngine::Layout::NCHW,
                                                           InferenceEngine::Layout::NHWC};
// Output layout is similarly limited as the input layout
const std::vector<InferenceEngine::Layout> outputLayouts = {InferenceEngine::Layout::NCHW,
                                                            InferenceEngine::Layout::NHWC};

INSTANTIATE_TEST_SUITE_P(smoke_precommit_VPU3720_BehaviorTestsPreprocess, InferRequestPreprocessTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_VPU3720_Auto_BehaviorTestsPreprocess, InferRequestPreprocessTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(CommonTestUtils::DEVICE_AUTO),
                                            ::testing::ValuesIn(autoConfigs)),
                         InferRequestPreprocessTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_precommit_VPU3720_BehaviorTestsPreprocess, InferRequestPreprocessConversionTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions), ::testing::ValuesIn(inputPrecisions),
                                            ::testing::ValuesIn(outputPrecisions), ::testing::ValuesIn(netLayouts),
                                            ::testing::ValuesIn(inputLayouts), ::testing::ValuesIn(outputLayouts),
                                            ::testing::Bool(), ::testing::Bool(),
                                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                                            ::testing::ValuesIn(configs)),
                         InferRequestPreprocessConversionTest::getTestCaseName);
}  // namespace
