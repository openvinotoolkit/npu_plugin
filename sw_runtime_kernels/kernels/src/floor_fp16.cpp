//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//

#include <param_floor.h>

#define VECTOR_SIZE 8     // Changes to this should be reflected in the code as well.
#define FP16_BIAS 15      /* expo bias */
#define FP16_TOTALBITS 16 /* total number of bits */
#define FP16_FRACTBITS 10 /* number of explicit fraction bits */
#define FP16_GREATINT ((FP16_BIAS + FP16_FRACTBITS) << FP16_FRACTBITS) /* big: all equal or above are integers */
#define FP16_TRUNCFRACT (((unsigned)-1 << FP16_FRACTBITS)) /* mask to truncate fraction bits at the binary point */

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {
static inline half8 floorh8(half8 x) {
    const short8 signMask = (short8)(1 << (FP16_TOTALBITS - 1));
    const half8 ones = (half8)1.0f;

    short8 xInt = (short8)x;
    short8 xSign = (xInt & signMask);
    short8 xAbs = (xInt & ~signMask);

    short8 xExpo = (xAbs >> FP16_FRACTBITS) - (short8)FP16_BIAS;
    short8 truncMask = ((short8)FP16_TRUNCFRACT >> xExpo);

    short8 isSign = (xInt >> (FP16_TOTALBITS - 1));
    short8 isZero = (xAbs - (short8)1) >> (FP16_TOTALBITS - 1);
    short8 isSmall = (xAbs - (short8)ones) >> (FP16_TOTALBITS - 1);
    isSmall &= ~isZero;
    short8 isGreat = ((short8)(FP16_GREATINT - 1) - xAbs) >> (FP16_TOTALBITS - 1);
    short8 isExact = (isGreat | isZero);
    short8 isFloor = ~(isExact | isSmall);

    short8 xTrunc = (xAbs & truncMask);
    short8 isInexact = (short8)(xTrunc != xAbs);

    short8 xFloor = (short8)((half8)xTrunc + (half8)((isSign & isInexact) & (short8)ones));
    short8 xSmall = (short8)((isSign & isSmall) & (short8)ones);

    half8 res = (half8)((isExact & xInt) | (xSign | ((isFloor & xFloor) | (isSmall & xSmall))));

    return res;
}

void floor_fp16(const struct FloorParams* lParams) {
    half8* p_act_data_v = (half8*)(lParams->input.dataAddr);
    half8* p_act_out_v = (half8*)(lParams->output.dataAddr);

    half* p_act_data_s = (half*)(lParams->input.dataAddr);
    half* p_act_out_s = (half*)(lParams->output.dataAddr);

    int32_t* pDims = (int32_t*)(lParams->input.dimsAddr);

    int32_t nElements = 1;
    int32_t i = 0;

    for (i = 0; i != lParams->input.numDims; i++) {
        nElements *= pDims[i];
    }
    const int numVectors = nElements / VECTOR_SIZE;

    for (i = 0; i < numVectors; i++) {
        p_act_out_v[i] = floorh8(p_act_data_v[i]);
    }

    for (i = numVectors * VECTOR_SIZE; i < nElements; i++) {
        p_act_out_s[i] = (half)(__builtin_floorf((p_act_data_s[i])));
    }
}
}
}  // namespace shave_lib
}  // namespace nn
