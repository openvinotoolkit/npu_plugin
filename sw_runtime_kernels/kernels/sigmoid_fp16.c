//#include "../asds/dpu2p7_descriptor.h"

#include <math.h>
#include <moviVectorUtils.h>
#include <param_sigmoid.h>

#define USE __attribute__((used))
//#define ARG_DATA __attribute__((section(".arg.data")))

//USE cfg_dpu_description descriptor_argument;

//__attribute__((section(".uuid.rodata")))
//const uint64_t uuid = 0x05063896D66C489D;

// This is how the activation SHAVE kernel will be called
// (kr->kernelEntry_)(ki->kernelArgs_.args_, ki->kernelArgs_.numArgs_);
void sigmoid_fp16(const struct SigmoidParams *lParams) {

    half* p_act_data = (half*)(lParams->input.dataAddr); // 0x1F000000
    half* p_act_out = (half*)(lParams->output.dataAddr); // 0x1F004000

    int32_t *pDims     = (int32_t *)((uint8_t*)(lParams) + lParams->input.dimsAddr); // 0x1F000000 + dimsAddr
    int64_t *pStrdides = (int64_t *)((uint8_t*)(lParams) + lParams->input.stridesAddr); // 0x1F000000 + stridesAddr

    int32_t nElements = 1;
    int32_t i = 0;
    half act = 0;

    for (i = 0; i!= lParams->input.numDims; i++ ) {
        // TODO: where pointers patch should be???
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
