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
#include <param_divide.h>

using namespace sw_params;

namespace nn {
namespace shave_lib {

extern "C" {

void divide_fp16(const struct DivideParams* lParams) {
    half* p_act_data1_s = (half*)(lParams->input1.dataAddr);
    half* p_act_data2_s = (half*)(lParams->input2.dataAddr);

    half* p_act_out_s = (half*)(lParams->output.dataAddr);

    int32_t* pDims = (int32_t*)(lParams->input1.dimsAddr);

    int32_t nElements = 1;
    int32_t i = 0;

    for (i = 0; i != lParams->input1.numDims; i++) {
        nElements *= pDims[i];
    }

    for (i = 0; i < nElements; i++) {
        p_act_out_s[i] = p_act_data1_s[i] / p_act_data2_s[i];
    }
}
}
}  // namespace shave_lib
}  // namespace nn
