// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <vpu/kmb_plugin_config.hpp>
#include <vpu/vpu_compiler_config.hpp>
#include "behavior/infer_request_config.hpp"
#include "ie_plugin_config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32
};

const std::vector<std::map<std::string, std::string>> configs = {};

const std::vector<std::map<std::string, std::string>> Inconfigs = {
    {},
    {{InferenceEngine::VPUConfigParams::KEY_VPU_COPY_OPTIMIZATION, InferenceEngine::PluginConfigParams::NO}},
    {{InferenceEngine::VPUConfigParams::KEY_VPU_IGNORE_UNKNOWN_LAYERS, InferenceEngine::PluginConfigParams::YES}},
    {{InferenceEngine::VPUConfigParams::KEY_VPU_HW_STAGES_OPTIMIZATION, InferenceEngine::PluginConfigParams::YES}},
    {{InferenceEngine::VPUConfigParams::KEY_VPU_NONE_LAYERS, "Tile"}},
    {{InferenceEngine::VPUConfigParams::KEY_VPU_NUMBER_OF_SHAVES, "5"}, {InferenceEngine::VPUConfigParams::KEY_VPU_NUMBER_OF_CMX_SLICES, "5"}},
    {{InferenceEngine::VPUConfigParams::KEY_VPU_HW_INJECT_STAGES, "YES"}},
    {{InferenceEngine::VPUConfigParams::KEY_VPU_HW_POOL_CONV_MERGE, "YES"}},
    {{"VPU_KMB_PREPROCESSING_SHAVES", "6"}},
    {{"VPU_KMB_PREPROCESSING_LPI", "8"}},
    {{VPU_COMPILER_CONFIG_KEY(ELTWISE_SCALES_ALIGNMENT), InferenceEngine::PluginConfigParams::YES}},
    {{VPU_COMPILER_CONFIG_KEY(CONCAT_SCALES_ALIGNMENT), InferenceEngine::PluginConfigParams::YES}},
    {{VPU_COMPILER_CONFIG_KEY(WEIGHTS_ZERO_POINTS_ALIGNMENT), InferenceEngine::PluginConfigParams::YES}},
    {{"VPU_KMB_SIPP_OUT_COLOR_FORMAT", "RGB"}},
    {{"VPU_KMB_FORCE_NCHW_TO_NHWC", InferenceEngine::PluginConfigParams::YES}}
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferConfigTests,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                            ::testing::ValuesIn(configs)),
                        InferConfigTests::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTests, InferConfigInTests,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(CommonTestUtils::DEVICE_KEEMBAY),
                            ::testing::ValuesIn(Inconfigs)),
                        InferConfigInTests::getTestCaseName);
}  // namespace
