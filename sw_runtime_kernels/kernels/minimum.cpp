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
#include <param_minimum.h>
#include <math.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void minimum(const struct MinimumParams *lParams) {
    half* in = (half*)lParams->input.dataAddr;
    half* in2 = (half*)lParams->input2.dataAddr;
    half* out = (half*)lParams->output.dataAddr;

    int32_t *pDims = (int32_t *)(lParams->input.dimsAddr);
    int32_t nElements = 1;

    for (int32_t i = 0; i != lParams->input.numDims; i++) {
        nElements *=  pDims[i];
    }

    for (int32_t i = 0; i < nElements; i++) {
        out[i] = MIN(in[i] , in2[i]);
    }
}

}
}  // namespace shave_lib
}  // namespace nn
