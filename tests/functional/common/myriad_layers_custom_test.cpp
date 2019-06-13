// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

#define ERROR_BOUND 0

static void refShuffleChannel(const Blob::Ptr src,
                              Blob::Ptr dst,
                              int group, int isCHW) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    const uint16_t *src_data = src->buffer();
          uint16_t *dst_data = dst->buffer();
    ASSERT_NE(src_data, nullptr);
    ASSERT_NE(dst_data, nullptr);
    int32_t IW = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    get_dims(src, IW, IH, IC);

    int G = group;
    int CX = IC / G;
    int CY = G;

    for (int cy = 0; cy < CY; cy++) {
        for (int cx = 0; cx < CX; cx++) {
            for (int h = 0; h < IH; h++) {
                for (int w = 0; w < IW; w++) {
                    if (isCHW) {
                        dst_data[(cx*CY + cy)*IW*IH + h*IW + w] = src_data[(cy*CX + cx)*IW*IH + h*IW + w];
                    } else {
                        dst_data[(cx*CY + cy) + h*IW*IC + w*IC] = src_data[(cy*CX + cx) + h*IW*IC + w*IC];
                    }
                }
            }
        }
    }
}

PRETTY_PARAM(Group, int)

typedef myriadLayerTestBaseWithParam<std::tuple<Dims, Group, std::string>> myriadLayersTestsShuffleChannel_nightly;

TEST_P(myriadLayersTestsShuffleChannel_nightly, ShuffleChannel) {
    tensor_test_params dims  = std::get<0>(GetParam());
    int group                = std::get<1>(GetParam());
    std::string customConfig = std::get<2>(GetParam());

    if(!customConfig.empty() && !CheckMyriadX()) {
        GTEST_SKIP()<<"Custom layers for MYRIAD2 not supported";
    }
    _config[VPU_CONFIG_KEY(CUSTOM_LAYERS)] = customConfig;

    SetInputTensor(dims);
    SetOutputTensor(dims);

    std::map<std::string, std::string> params;
    params["group"] = std::to_string(group);

    ASSERT_NO_FATAL_FAILURE(NetworkInit("ShuffleChannel", &params, 0, 0, nullptr, Precision::FP16, Precision::FP16));

    ASSERT_TRUE(Infer());

    ASSERT_NO_FATAL_FAILURE(refShuffleChannel(_inputMap.begin()->second, _refBlob, group, false));

    Compare(_outputMap.begin()->second, _refBlob, ERROR_BOUND);
}

static std::vector<Dims> s_ShuffleChannelTensors = {
    {{1,  48, 28, 28}},
    {{1,  96, 14, 14}},
    {{1, 192,  7,  7}},
};

static std::vector<Group> s_ShuffleChannel_group = {
    2
};

static std::vector<std::string> s_MVNCustomConfig = {
    {TestsCommon::get_data_path() + "/vpu/mvcl/customLayers_m7.xml"}
};

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersTestsShuffleChannel_nightly,
        ::testing::Combine(
        ::testing::ValuesIn(s_ShuffleChannelTensors),
        ::testing::ValuesIn(s_ShuffleChannel_group),
        ::testing::ValuesIn(s_MVNCustomConfig)));
