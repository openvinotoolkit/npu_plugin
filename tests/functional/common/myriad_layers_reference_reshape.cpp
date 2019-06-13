// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/precision_utils.h>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include <single_layer_common.hpp>

using namespace InferenceEngine;

void ref_reshape_wrap(InferenceEngine::Blob::Ptr src,
                      InferenceEngine::Blob::Ptr dst,
                      const ParamsStruct& params) {
    //ASSERT_TRUE(params.empty());
    ref_reshape(src, dst);
}

void ref_reshape(const Blob::Ptr src,
                 Blob::Ptr dst) {

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

    ASSERT_EQ(IW * IH *IC, OW * OH * OC);

    const uint16_t *src_data = src->buffer();
    uint16_t *dst_data = dst->buffer();
    size_t sz = IW * IH *IC;
    std::vector<uint16_t> temp(sz);
    uint16_t* pTmp = temp.data();
    //HWC->CHW
    for (size_t ic = 0; ic < IC; ++ic) {
        for (size_t ih = 0; ih < IH; ++ih) {
            for (size_t iw = 0; iw < IW; ++iw) {
                size_t iidx = iw + IW * ( ih  + ic * IH );
                size_t oodx = ic + IC * ( iw  + ih * IW );
                temp[iidx] = src_data[oodx];
            }
        }
    }
    //CHW->HWC
    for (size_t ow = 0; ow < OW; ++ow) {
        for (size_t oh = 0; oh < OH; ++oh) {
            for (size_t oc = 0; oc < OC; ++oc) {
                size_t iidx = ow + OW * ( oh  + oc * OH );
                size_t oodx = oc + OC * ( ow  + oh * OW );
                dst_data[oodx] = temp[iidx];
            }
        }
    }
}
