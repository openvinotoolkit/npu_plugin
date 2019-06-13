// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"
#include <algorithm>

using std::tr1::tuple;
using std::tr1::get;

using namespace InferenceEngine;

struct nd_tensor_test_params {
    SizeVector dims;
    int axis;
};

PRETTY_PARAM(test_params, nd_tensor_test_params);
PRETTY_PARAM(tiles, int);

bool iter(SizeVector& in, SizeVector& out)
{
    bool flag = true;
    for(int i = 0; i < out.size(); i++) {
        if(in[i] < out[i] - 1) {
            in[i]++;
            break;
        } else {
            if(i == out.size() - 1) {
                flag = false;
                break;
            }
            in[i] = 0;
        }
    }
    return flag;
}

void calcPos(SizeVector& in, SizeVector& out)
{
    for(int i = 0; i < out.size(); i++) {
        out[i] %= in[i];
    }
}

int calcOffset(SizeVector& in, SizeVector& out)
{
    int offset = in[0];
    for(int i = 1; i < in.size(); i++) {
        int mul = in[i];
        for(int j = i - 1; j >= 0; j--)
            mul *= out[j];
        offset += mul;
    }
    return offset;
}

void ref_tile(const InferenceEngine::Blob::Ptr src,
              InferenceEngine::Blob::Ptr dst,
              int axis_val,
              int tiles_val)
{
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);

    SizeVector in_size;
    SizeVector out_size;
    in_size = src->dims();
    out_size = dst->dims();
    Layout layout = src->layout();
    SizeVector curr_size(in_size.size());
    const uint16_t *src_data = src->buffer();
    uint16_t *dst_data = dst->buffer();

// TODO: investigate this case
    if (layout == NCHW || layout == NHWC) {
        size_t N = in_size[3];
        size_t C = in_size[2];
        size_t H = in_size[1];
        size_t W = in_size[0];

        size_t N1 = out_size[3];
        size_t C1 = out_size[2];
        size_t H1 = out_size[1];
        size_t W1 = out_size[0];
        for (size_t n = 0; n < N1; n++) {
            for (size_t c = 0; c < C1; c++) {
                for (size_t h = 0; h < H1; h++) {
                    for (size_t w = 0; w < W1; w++) {
                        size_t idx = layout == NCHW ?
                                           (w % W) + (h % H) * W + (c % C) * W * H + (n % N) * W * H * C : 
                                           (c % C) + (w % W) * C + (h % H) * C * W + (n % N) * W * H * C;
                        size_t actualIdx = layout == NCHW ?
                                           w + h * W1 + c * W1 * H1 + n * W1 * H1 * C1 : 
                                           c + w * C1 + h * C1 * W1 + n * W1 * H1 * C1;
                        dst_data[actualIdx] = src_data[idx];
                    }
                }
            }
        }
    } else {
        do {
            SizeVector ref = curr_size;
            calcPos(in_size, ref);
            dst_data[calcOffset(curr_size, out_size)] = src_data[calcOffset(ref, in_size)];
        } while(iter(curr_size, out_size));
    }
}

typedef myriadLayerTestBaseWithParam<tuple<test_params, tiles>> myriadLayerTestTile_nightly;

TEST_P(myriadLayerTestTile_nightly, Tile) {
    _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    _config[VPU_CONFIG_KEY(COMPUTE_LAYOUT)] = VPUConfigParams::VPU_NCHW;

    nd_tensor_test_params input_dims = get<0>(GetParam());
    int tiles = get<1>(GetParam());
    int ndims = input_dims.dims.size();
    int axis = input_dims.axis;
    auto dims = input_dims.dims;
    SetInputTensors({dims});
    dims[axis] *= tiles;
    SetOutputTensors({dims});
    std::map<std::string, std::string> params;
    params["axis"] = std::to_string(axis);
    params["tiles"] = std::to_string(tiles);

    NetworkInit("Tile", &params, 0, 0, nullptr, InferenceEngine::Precision::FP16);
    auto inputBlob = _inputMap.begin()->second;
    auto outputBlob = _outputMap.begin()->second;
    SetFirstInputToRange(1.0f, 100.0f);
    ASSERT_TRUE(Infer());
    ref_tile(inputBlob, _refBlob, axis, tiles);
    Compare(outputBlob, _refBlob, 0);
}

INSTANTIATE_TEST_CASE_P(DISABLED_accuracyAdd, myriadLayerTestTile_nightly,
        ::testing::Combine(
            ::testing::Values<test_params>(
                                     MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6}, 0)
                                   , MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6, 7}, 0)
                                   , MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6, 7}, 1)
                                   , MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6, 27, 13}, 0)
                                   , MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6, 27, 13}, 1)
                                   , MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6, 27, 13}, 2)
                                   , MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6, 27, 13, 18}, 0)
                                   , MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6, 27, 13, 18}, 1)
                                   , MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6, 27, 13, 18}, 2)
                                   , MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6, 27, 13, 18}, 3))

          , ::testing::Values<tiles>(2, 3, 5)
                        ));

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayerTestTile_nightly,
        ::testing::Combine(
            ::testing::Values<test_params>(
                                     MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6}, 1)
                                   , MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6}, 2)
                                   , MAKE_STRUCT(nd_tensor_test_params, {5, 6}, 0)
                                   , MAKE_STRUCT(nd_tensor_test_params, {5, 6}, 1)
                                   , MAKE_STRUCT(nd_tensor_test_params, {6}, 0)
                                   , MAKE_STRUCT(nd_tensor_test_params, {6, 5, 6, 7}, 2)
                                   , MAKE_STRUCT(nd_tensor_test_params, {6, 5, 6, 7}, 3)
                                   , MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6, 27, 13}, 3)
                                   , MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6, 27, 13}, 4)
                                   , MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6, 27, 13, 18}, 4)
                                   , MAKE_STRUCT(nd_tensor_test_params, {4, 5, 6, 27, 13, 18}, 5))

          , ::testing::Values<tiles>(2, 3, 5)
                        ));
