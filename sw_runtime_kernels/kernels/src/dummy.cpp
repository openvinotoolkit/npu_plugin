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

#include <param_dummy.h>
#include <limits.h>

#ifdef __cplusplus
using namespace sw_params;
extern "C" {
#endif

void dummy(const struct DummyParams *lParams) {

    const struct MemRefData * inputs  = lParams->tensors;
    const struct MemRefData * outputs = lParams->tensors + lParams->numIns;

    uint8_t data[MAX_KERNEL_INPUTS];
    int saved = 0;
    for (int i = 0; i < lParams->numIns; i++) {
        int32_t *pDims = (int32_t *)(inputs[i].dimsAddr);
        int64_t *pStrides = (int64_t *)(inputs[i].stridesAddr);
        int32_t numDims = inputs[i].numDims;
        if (numDims > 0 && pDims[numDims - 1] * pStrides[numDims - 1] / CHAR_BIT / sizeof(uint8_t) > 0) {
            uint8_t* p_act_data_s = (uint8_t*)(inputs[i].dataAddr);
            data[saved++] = p_act_data_s[0];
        }
    }  // As a result data contains the first bytes from each input tensor

    // Save the data (or only part up to the output tensor size) into each output tensor
    for (int i = 0; i < lParams->numOuts; i++) {
        int32_t *pDims = (int32_t *)(outputs[i].dimsAddr);
        int64_t *pStrides = (int64_t *)(outputs[i].stridesAddr);
        int32_t numDims = outputs[i].numDims;
        if (numDims > 0 && pDims[numDims - 1] * pStrides[numDims - 1] / CHAR_BIT / sizeof(uint8_t) > 0) {
            uint8_t* p_act_data_s = (uint8_t*)(outputs[i].dataAddr);
            int toSave = pDims[numDims - 1] * pStrides[numDims - 1] / CHAR_BIT / sizeof(uint8_t);
            toSave = (saved > toSave) ? toSave : saved;
            for (int j = 0; j < toSave; j++) {
                p_act_data_s[j] = data[j];
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
