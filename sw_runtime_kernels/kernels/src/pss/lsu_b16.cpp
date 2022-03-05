//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <moviVectorTypes.h>
#include <math.h>
#include <pss/param_lsu_b16.h>

using namespace sw_params;

extern "C"
void lsu_b16(const struct LsuB16Params *lParams) {
    const struct MemRefData* inputs = lParams->tensors + 0;
    const struct MemRefData* outputs = lParams->tensors + LsuB16Params::NumInputs;

    const int32_t *dims = (int32_t*)(outputs[0].dimsAddr);

    int32_t nElements = 1;

    for (int32_t i = 0; i < outputs[0].numDims; ++i) {
        // TODO: check overflow
        nElements *= dims[i];
    }

    half* in = (half*)(inputs[0].dataAddr);
    half* out = (half*)(outputs[0].dataAddr);

    for (int32_t e = 0; e < nElements; ++e) {
        float val = __builtin_shave_lsu_ld32_b16_f32_r(in++);
        __builtin_shave_lsu_st16_f32_b16_rr(val, out++);
    }
}
