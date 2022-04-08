//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <moviVectorUtils.h>
#include <param_erf.h>

#define FP16_TOTALBITS 16 /* total number of bits */

#define VECTOR_SIZE (8) /* Changes to this should be reflected in the code as well */

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

// calculate erf of 8 elements (half type) in parallel
inline half8 erfh8(half8 x) {
    const short8 signMask = (short8)(1 << (FP16_TOTALBITS - 1));
    const half8 clipBound = (half8)2.86f;

    const half8 xAbs = (half8)(~signMask & (short8)x);
    const short8 xSign = (signMask & (short8)x);

    //  Points clip_bound and -clip_bound are extremums for this polynom
    //  So in order to provide better accuracy comparing to std::erf we have to clip input range
    const short8 isBig = (xAbs > clipBound);  // return +-1;
    const short8 inRange = ~isBig;

    const half8 one = (half8)1.0f;
    const half8 big = (half8)(xSign | (short8)one);

    //  A polynomial approximation of the error function
    const float8 erfNumerator[4] = {(float8)90.0260162353515625f, (float8)2232.00537109375f, (float8)7003.3251953125f,
                                    (float8)55592.30078125f};
    const float8 erfDenominator[5] = {(float8)33.56171417236328125f, (float8)521.35797119140625f,
                                      (float8)4594.32373046875f, (float8)22629.0f, (float8)49267.39453125f};

    const float8 xf = mvuConvert_float8(x);
    const float8 x2 = xf * xf;

    float8 num = (float8)9.60497379302978515625f;
    for (const auto c : erfNumerator)
        num = num * x2 + c;
    num *= xf;

    float8 den = (float8)1.0f;
    for (const auto c : erfDenominator)
        den = den * x2 + c;

    half8 res = mvuConvert_half8(num / den);

    return (half8)((isBig & (short8)big) | (inRange & (short8)res));
}

void erf_fp16(const struct ErfParams* lParams) {
    half8* pActDataV = (half8*)(lParams->input.dataAddr);
    half8* pActOutV = (half8*)(lParams->output.dataAddr);

    half* pActDataS = (half*)(lParams->input.dataAddr);
    half* pActOutS = (half*)(lParams->output.dataAddr);

    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);

    int32_t numberOfElements = 1;
    int32_t i = 0;

    for (i = 0; i != lParams->input.numDims; i++) {
        numberOfElements *= pDims[i];
    }

    const int numberOfVectors = numberOfElements / VECTOR_SIZE;

#pragma clang loop unroll_count(8)
    for (i = 0; i < numberOfVectors; i++) {
        half8 vIn = pActDataV[i];
        half8 vOut;
        vOut = erfh8(vIn);
        pActOutV[i] = vOut;
    }

    int32_t numberOfTailElements = numberOfElements - numberOfVectors * VECTOR_SIZE;
    if (numberOfTailElements > 0) {
        half8 tailIn;
        half8 tailOut;
        half* p = (half*)&pActDataV[numberOfVectors];
        //  read tailing elements that less than VECTOR_SIZE
        for (int i = 0; i < numberOfTailElements; i++) {
            tailIn[i] = p[i];
        }

        tailOut = erfh8(tailIn);

        for (int i = 0; i < numberOfTailElements; i++) {
            pActOutS[numberOfVectors * VECTOR_SIZE + i] = tailOut[i];
        }
    }
}

}  // extern "C"

}  // namespace shave_lib
}  // namespace nn
