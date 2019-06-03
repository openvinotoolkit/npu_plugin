// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_layers_tests.hpp"

using namespace InferenceEngine;

static void refPermute(const Blob::Ptr src, Blob::Ptr dst, const int* order_hwc, tensor_test_params &ref_param) {

    const ie_fp16 *srcData = src->cbuffer().as<ie_fp16*>();
    ie_fp16 *dstData = dst->buffer().as<ie_fp16 *>();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);

    int input_dim[3];
    size_t IW = src->dims()[0];
    size_t IH = src->dims()[1];
    size_t IC = src->dims()[2];

    input_dim[0] = static_cast<int>(IH);
    input_dim[1] = static_cast<int>(IW);
    input_dim[2] = static_cast<int>(IC);

    int out_height    = input_dim[order_hwc[0]];
    int out_width     = input_dim[order_hwc[1]];
    int out_channels  = input_dim[order_hwc[2]];

    ref_param.c = out_channels;
    ref_param.h = out_height;
    ref_param.w = out_width;

    int out_steps[3];
    out_steps[order_hwc[0]] = out_width * out_channels;
    out_steps[order_hwc[1]] = out_channels;
    out_steps[order_hwc[2]] = 1;

    for( size_t i = 0; i < IH; i++) {
        for(size_t j = 0; j < IW; j++) {
            for(size_t k = 0; k < IC ; k++) {
                int iidx = k + j * IC +  i * IW *IC ;
                int oidx = i * out_steps[0] + j * out_steps[1] + k * out_steps[2];
                dstData[oidx] = srcData[iidx];
            }
        }
    }
}

struct offset_test_params {
    size_t order0;
    size_t order1;
    size_t order2;
    size_t order3;
};

PRETTY_PARAM(Offsets, offset_test_params);

// Show contents of offset test param by not hexadecimal but integer
static inline void PrintTo(const offset_test_params& param, ::std::ostream* os)
{
    *os << "{ " << param.order0 << ", " << param.order1 << ", " << param.order2 << ", " << param.order3 << "}";
}
typedef std::tuple<InferenceEngine::SizeVector, InferenceEngine::SizeVector> PermuteParams;

class myriadLayersPermuteTests_nightly: public myriadLayersTests_nightly, /*input tensor, order */
                                        public testing::WithParamInterface<PermuteParams> {
};

static void genRefData(InferenceEngine::Blob::Ptr blob) {
    ASSERT_NE(blob, nullptr);
    Layout layout = blob->layout();
    SizeVector dims = blob->getTensorDesc().getDims();

    ie_fp16* ptr = blob->buffer().as<ie_fp16*>();
    if (layout == NCHW || layout == NHWC) {
        size_t N = dims[0];
        size_t C = dims[1];
        size_t H = dims[2];
        size_t W = dims[3];
        float counter = 0.f;
        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        size_t actualIdx = layout == NCHW ?
                                           w + h * W + c * W * H + n * W * H * C : c + w * C + h * C * W +
                                                                                   n * W * H * C;
                        ptr[actualIdx] = PrecisionUtils::f32tof16(counter);
                        counter += 0.25f;
                    }
                }
            }
        }
    } else {
        ASSERT_TRUE(false);
    }
}

TEST_P(myriadLayersPermuteTests_nightly, Permute) {
    std::map<std::string, std::string> params;
    InferenceEngine::SizeVector output_tensor;
    int32_t IW = 0;
    int32_t IH = 0;
    int32_t IC = 0;
    int32_t I_N = 0;
    size_t  group = 0;

    auto p = ::testing::WithParamInterface<PermuteParams>::GetParam();
    auto input_tensor = std::get<0>(p);
    auto order =        std::get<1>(p);
    get_dims(input_tensor, IW, IH, IC, I_N);
    if (I_N > 1)
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(NO);
    else
        _config[VPU_CONFIG_KEY(DETECT_NETWORK_BATCH)] = CONFIG_VALUE(YES);
    if (input_tensor.size()) {
        gen_dims(output_tensor, input_tensor.size(), input_tensor[order[3]],
                                                     input_tensor[order[2]], 
                                                     input_tensor[order[1]], 
                                                     input_tensor[order[0]]);
    }
    std::string orderStr;
    for (int i = 0; i < order.size() - 1; ++i) {
        orderStr += std::to_string(order[i]);
        orderStr += ",";
    }
    if (!order.empty()) {
        orderStr += std::to_string(order.back());
    }
    std::map<std::string, std::string> layer_params = {
              {"order", orderStr}
    };
    _genDataCallback = genRefData;
    AddLayer("Permute",
             &layer_params,
             {input_tensor},
             {output_tensor},
             ref_permute_wrap);
    ASSERT_TRUE(GenerateNetAndInfer(CheckMyriadX(), true, 3));
    Compare(_outputMap.begin()->second, GenReferenceOutput(), 0.0f);
}

static const std::vector<InferenceEngine::SizeVector> s_inTensors = {
    {1, 36, 19, 19},
    {1, 2, 7, 8},
    {1, 196, 12, 2}
};

static const std::vector<InferenceEngine::SizeVector> s_permuteTensors = {
    {0, 1, 2, 3},
    {0, 1, 3, 2},
    {0, 2, 1, 3},
    {0, 2, 3, 1},
    {0, 3, 1, 2},
    {0, 3, 2, 1}
};

INSTANTIATE_TEST_CASE_P(accuracy, myriadLayersPermuteTests_nightly,
        ::testing::Combine(
            ::testing::ValuesIn(s_inTensors)
          , ::testing::ValuesIn(s_permuteTensors)
));

INSTANTIATE_TEST_CASE_P(accuracyFasterRCNN, myriadLayersPermuteTests_nightly,
        ::testing::Combine(
             ::testing::Values<InferenceEngine::SizeVector>({1, 24, 14, 14})
            ,::testing::Values<InferenceEngine::SizeVector>({0, 2, 3, 1})
            ));


