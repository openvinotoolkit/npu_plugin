// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <inference_engine/precision_utils.h>
#include "myriad_layers_tests.hpp"
#include "myriad_layers_reference_functions.hpp"

using namespace InferenceEngine;

void ref_eltwise(const Blob::Ptr src1,
                 const Blob::Ptr src2,
                 Blob::Ptr dst,
                 eltwise_kernel fun,
                 std::vector<float> coeff) {
    ASSERT_NE(src1, nullptr);
    ASSERT_NE(src2, nullptr);
    ASSERT_NE(dst, nullptr);
    uint16_t *dstData = dst->buffer().as<uint16_t*>();
    uint16_t *src1Data = src1->buffer().as<uint16_t*>();
    uint16_t *src2Data = src2->buffer().as<uint16_t*>();

    ASSERT_NE(src1Data, nullptr);
    ASSERT_NE(src2Data, nullptr);
    ASSERT_NE(dstData, nullptr);

    for (int i = 0; i < dst->size(); i++) {
        float val = fun(PrecisionUtils::f16tof32(src1Data[i])*coeff[0], PrecisionUtils::f16tof32(src2Data[i])*coeff[1]);
        dstData[i] = PrecisionUtils::f32tof16(val);
    }
}
