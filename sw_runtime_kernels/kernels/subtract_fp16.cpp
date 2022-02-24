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

#ifdef CONFIG_HAS_LRT_SRCS
#include <nn_log.h>
#else
#define nnLog(level, ...)
#endif
#include <param_subtract.h>

#define VECTOR_SIZE (8) /* Changes to this should be reflected in the code as well */

#define intrinsic_vau_vec(intrinsic, vin1, vin2, vout) \
    (vout) = intrinsic((vin1),(vin2))

#define sub_vec(vin1, vin2, vout) (intrinsic_vau_vec(__builtin_shave_vau_sub_f16_rr, vin1, vin2, vout))

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void subtract_fp16(uint32_t lParamsAddr) {

    const SubtractParams* lParams = (const SubtractParams*)lParamsAddr;

    half8* in = (half8*)(lParams->input.dataAddr);
    half8* in2 = (half8*)(lParams->input2.dataAddr);
    half8* out = (half8*)(lParams->output.dataAddr);

    int32_t *pDims = (int32_t *)(lParams->input.dimsAddr);

    int32_t nElements = 1;
    int32_t i = 0;

    for (i = 0; i!= lParams->input.numDims; i ++ ) {
        nElements *=  pDims[i];
    }
    const int numVectors = nElements / VECTOR_SIZE;

#pragma clang loop unroll_count(8)
    for (i = 0; i < numVectors; i ++) {
        half8 vin1 = in[i];
        half8 vin2 = in2[i];
        half8 vout;
        sub_vec(vin1, vin2, vout);
        out[i] = vout;
    }

}

}
}  // namespace shave_lib
}  // namespace nn
