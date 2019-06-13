// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/precision_utils.h>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

using namespace InferenceEngine;

void ref_PReLU(const Blob::Ptr src,
               Blob::Ptr dst,
               const uint16_t *weights,
               size_t weightsSize) {
    ASSERT_EQ(src->dims().size(), dst->dims().size());
    ie_fp16 *srcData = static_cast<ie_fp16*>(src->buffer());
    ie_fp16 *dstData = static_cast<ie_fp16*>(dst->buffer());
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);
    // dst = max(src, 0) + w * min(src, 0)
    for (size_t indx = 0; indx < src->size(); indx++) {
        float w = PrecisionUtils::f16tof32(weights[indx % weightsSize]);
        float src = PrecisionUtils::f16tof32(srcData[indx]);
        float dst = std::max(src, 0.f) + w * std::min(src, 0.f);
        dstData[indx] = PrecisionUtils::f32tof16(dst);
    }
}

void ref_PReLU_wrap(const InferenceEngine::Blob::Ptr src,
                    InferenceEngine::Blob::Ptr dst,
                    const uint16_t *weights,
                    size_t weightsSize,
                    size_t biasSize,
                    const ParamsStruct& params) {
    int channel_shared = 0;
    if (!params.empty()) {
        auto iter = params.find(PRELU_PARAM);
        if (iter != params.end()) {
             channel_shared = std::stoi(iter->second);
        }
    }

    size_t get_weightsSize = 1;
    if (channel_shared == 0) {
        if (src->dims().size() == 2) {
            get_weightsSize = src->dims()[0];
        } else {
            int32_t OW = 0;
            int32_t OH = 0;
            int32_t OC = 0;
            get_dims(src, OW, OH, OC);
            get_weightsSize = OC;
        }
    }
    ASSERT_EQ(get_weightsSize, weightsSize);
    ref_PReLU(src, dst, weights, weightsSize);
}
