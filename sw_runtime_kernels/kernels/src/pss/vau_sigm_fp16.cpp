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
#include <pss/param_vau_sigm.h>

using namespace sw_params;

extern "C"
void vau_sigm_fp16(const struct VauSigmParams *lParams) {
    const struct MemRefData* inputs = lParams->tensors + 0;
    const struct MemRefData* outputs = lParams->tensors + VauSigmParams::NumInputs;

    const int32_t *dims = (int32_t*)(outputs[0].dimsAddr);

    int32_t nElements = 1;

    for (int32_t i = 0; i < outputs[0].numDims; ++i) {
        // TODO: check overflow
        nElements *= dims[i];
    }

    // NOTE: test must align tensor size according to vector size
    nElements = nElements / 8;

    half8* in = (half8*)(inputs[0].dataAddr);
    half8* out = (half8*)(outputs[0].dataAddr);

    for (int32_t e = 0; e < nElements; ++e) {
        half8 a = *in++;
        *out++ = __builtin_shave_vau_sigm_v8f16_r(a);
    }
}
