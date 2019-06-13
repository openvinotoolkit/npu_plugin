// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/precision_utils.h>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

using namespace InferenceEngine;

void ref_permute_wrap(const InferenceEngine::Blob::Ptr src,
                 InferenceEngine::Blob::Ptr dst,
                 const ParamsStruct& params) {
    std::vector<int> order;
    if (!params.empty()) {
        auto iter = params.find("order");
        if (iter != params.end()) {
            std::string param = iter->second;
            auto pos = std::string::npos;
            do {
                pos = param.find_first_of(",");
                if (pos == std::string::npos) {
                    if (!param.empty())
                        order.push_back(std::stoi(param));
                    break;
                }
                std::string val = param.substr(0, pos);
                order.push_back(std::stoi(val));
                param = param.substr(pos + 1, param.size() - 1);
            }while(pos != std::string::npos);
        }
    }
    int nhwc_to_nchw[4] = {0, 2, 3, 1};
    int nchw_to_nhwc[4] = {0, 3, 1, 2};
    int ref_order[4]; // NCHW
    for (int i = 0; i < 4; i++) {
        ref_order[i] = nchw_to_nhwc[order[nhwc_to_nchw[i]]];
    }

    ref_Permute(src, dst, ref_order);
}

void ref_Permute(const Blob::Ptr src, Blob::Ptr dst, const int* order) {
    const ie_fp16 *srcData = src->cbuffer().as<ie_fp16*>();
    ie_fp16 *dstData = dst->buffer().as<ie_fp16 *>();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);
    ASSERT_EQ(src->getTensorDesc().getLayout(), dst->getTensorDesc().getLayout());
    int input_dim[4];
    int32_t OW = 1;
    int32_t OH = 1;
    int32_t OC = 1;
    int32_t ON = 1;
    int32_t IH = 1;
    int32_t IC = 1;
    int32_t IW = 1;
    int32_t I_N = 1;
    get_common_dims(*src, IW, IH, IC, I_N);
    get_common_dims(*dst, OW, OH, OC, ON);
    input_dim[0] = I_N;
    input_dim[1] = IC;
    input_dim[2] = IH;
    input_dim[3] = IW;
    int order_hwc[3];
    order_hwc[0] = (order[1] - 1);
    order_hwc[1] = (order[2] - 1);
    order_hwc[2] = (order[3] - 1);

    input_dim[0] = IH;
    input_dim[1] = IW;
    input_dim[2] = IC;

    int out_height    = input_dim[order_hwc[0]];
    int out_width     = input_dim[order_hwc[1]];
    int out_channels  = input_dim[order_hwc[2]];

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
