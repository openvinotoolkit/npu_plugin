// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <inference_engine/precision_utils.h>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

using namespace InferenceEngine;

const std::string relu_param = "negative_slope";

void ref_ReLU(Blob::Ptr inTensor,
              Blob::Ptr outTensor,
              float negative_slope) {
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
        float val = std::max(inpt, 0.0f) + negative_slope * std::min(inpt, 0.0f);
        blobOutDataFp16[indx] = PrecisionUtils::f32tof16(val);
    }
}

void ref_ReLU_wrap(InferenceEngine::Blob::Ptr inTensor,
              InferenceEngine::Blob::Ptr outTensor,
              const ParamsStruct& params) {
    float negative_slope = 0.0f;
    if (!params.empty()) {
        auto iter = params.find(relu_param);
        if (iter != params.end()) {
            negative_slope = std::stof(iter->second);
        }
    }
    ref_ReLU(inTensor, outTensor, negative_slope);
}


