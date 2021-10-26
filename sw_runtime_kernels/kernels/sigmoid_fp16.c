// {% copyright %}

#include <moviVectorTypes.h>
#include <math.h>
#include <param_sigmoid.h>

#include <stdio.h>

void sigmoid_fp16(const struct SigmoidParams *lParams) {

    printf("I am in kernel\n");

    half* p_act_data = (half*)(lParams->input.dataAddr); // 0x1F000000
    half* p_act_out = (half*)(lParams->output.dataAddr); // 0x1F004000

    int32_t *pDims     = (int32_t *)((uint8_t*)(lParams) + lParams->input.dimsAddr);    // 0x1F000000 + dimsAddr
    int64_t *pStrdides = (int64_t *)((uint8_t*)(lParams) + lParams->input.stridesAddr); // 0x1F000000 + stridesAddr
    pDims = (int32_t *)(lParams->input.dimsAddr);

    printf("SHAVE: dataAddr = 0x%X\n", lParams->input.dataAddr);
    printf("SHAVE: dimsAddr = 0x%X\n", lParams->input.dimsAddr);
    printf("SHAVE: pDims = 0x%X\n", pDims);

    int32_t nElements = 1;
    int32_t i = 0;
    half act = 0;

    // for(i = 0; i < 30; i += 4)
    //     printf("SHAVE: lParams = 0x%X\n", *(uint32_t *)((uint8_t*)(lParams)+i));

    for (i = 0; i!= lParams->input.numDims; i++ ) {
        // TODO: where pointers patch should be???
        // TODO: check overflow
        nElements *=  pDims[i];
        printf("DEBUG: dims[%d] = 0x%X\n", i, pDims[i]);
    }


    printf("DEBUG: nElements = %d\n", nElements);

    for (uint32_t e = 0; e < nElements; ++e) {
        printf("DEBUG: %f\n", *p_act_data);
        act = *p_act_data++ * -1.0f;
        act = 1.0f + expf(act);
        act = 1.0f / act;
        *p_act_out++ = (half)act;
        // *p_act_out++ = (half)42;
    }
}
