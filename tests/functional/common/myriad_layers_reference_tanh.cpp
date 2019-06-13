// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/precision_utils.h>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"
#include <math.h>

using namespace InferenceEngine;

void ref_tanh_wrap(InferenceEngine::Blob::Ptr src,
                   InferenceEngine::Blob::Ptr dst,
                   const ParamsStruct& params) {
    ASSERT_TRUE(params.empty());
    ref_tanh(src, dst);
}

void ref_tanh(const InferenceEngine::Blob::Ptr src,
              InferenceEngine::Blob::Ptr dst) {
    ASSERT_NE(src, nullptr);
    ASSERT_NE(dst, nullptr);
    ASSERT_EQ(src->dims().size(), dst->dims().size());
    uint16_t *srcData = src->buffer();
    uint16_t *dstData = dst->buffer();
    ASSERT_NE(srcData, nullptr);
    ASSERT_NE(dstData, nullptr);
    for (size_t indx = 0; indx < src->size(); indx++) {
        dstData[indx] = 
            PrecisionUtils::f32tof16(tanh(PrecisionUtils::f16tof32(srcData[indx])));
    }
}
