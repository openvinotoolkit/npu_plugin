// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/precision_utils.h>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

#ifdef MAX
#undef MAX
#endif
#define MAX(a, b) ((a) > (b))?(a):(b)

#ifdef MIN
#undef MIN
#endif
#define MIN(a, b) ((a) < (b))?(a):(b)

using namespace InferenceEngine;

void ref_Clamp(Blob::Ptr inTensor,
              Blob::Ptr outTensor,
              float min,
              float max) {
    ASSERT_NE(inTensor, nullptr);
    ASSERT_NE(outTensor, nullptr);
    uint16_t *blobRawDataFp16 = inTensor->buffer();
    ASSERT_NE(blobRawDataFp16, nullptr);
    uint16_t *blobOutDataFp16 = outTensor->buffer();
    ASSERT_NE(blobOutDataFp16, nullptr);
    size_t count = inTensor->size();
    ASSERT_EQ(count, outTensor->size());
    for (size_t indx = 0; indx < count; ++indx) {
        float inpt = PrecisionUtils::f16tof32(blobRawDataFp16[indx]);
        float val = MIN(max, MAX(min, inpt));
        blobOutDataFp16[indx] = PrecisionUtils::f32tof16(val);
    }
}

void ref_Clamp_wrap(InferenceEngine::Blob::Ptr inTensor,
              InferenceEngine::Blob::Ptr outTensor,
              const ParamsStruct& params) {
    float min = 0.0f;
    float max = 6.0f;
    if (!params.empty()) {
        auto iter = params.find("max");
        if (iter != params.end()) {
            max = std::stof(iter->second);
        }
        iter = params.find("min");
        if (iter != params.end()) {
            min = std::stoi(iter->second);
        }
    }
    ref_Clamp(inTensor, outTensor, min, max);
}


