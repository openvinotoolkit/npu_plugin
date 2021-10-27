// {% copyright %}

#include <moviVectorTypes.h>
#include <math.h>
#include <param_sigmoid.h>

void sigmoid_fp16(uint32_t lParamsAddr, uint8_t * cmxData, int32_t availableCmxBytes) {

    const struct SigmoidParams * lParams = (const struct SigmoidParams*)lParamsAddr;

    half* p_act_data = (half*)(lParams->input.dataAddr); // 0x1F000000
    half* p_act_out = (half*)(lParams->output.dataAddr); // 0x1F004000

    int32_t *pDims     = (int32_t *)(lParams->input.dimsAddr);    // 0x1E000000 , dimsAddr is globally computed in  WIN_E
    int64_t *pStrdides = (int64_t *)(lParams->input.stridesAddr); // 0x1E000000 , stridesAddr is globally computed in  WIN_E

    int32_t nElements = 1;
    int32_t i = 0;
    half act = 0;

    for (i = 0; i!= lParams->input.numDims; i++ ) {
        // TODO: check overflow
        nElements *=  pDims[i];
    }

    for (uint32_t e = 0; e < nElements; ++e) {
        act = *p_act_data++ * -1.0f;
        act = 1.0f + expf(act);
        act = 1.0f / act;
        *p_act_out++ = (half)act;
    }
}
