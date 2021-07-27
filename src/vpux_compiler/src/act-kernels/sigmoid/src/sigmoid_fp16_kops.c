#include "../../asds/kernel_op.h"
#include <math.h>


#define USE __attribute__((used)) 
#define ARG_DATA __attribute__((section(".arg.data")))

ARG_DATA USE slf_kernel_asds descriptor_argument;

__attribute__((section(".uuid.rodata"))) 
const uint64_t uuid = 0x05063896D66C489E;

static uint32_t call_counter = 1;

// This is how the activation SHAVE kernel will be called
// (kr->kernelEntry_)(ki->kernelArgs_.args_, ki->kernelArgs_.numArgs_);
void sigmoid_fp16(void* args, uint32_t num_args) {
    slf_kernel_params* params = (slf_kernel_params*)args;
    const kernel_op* p_kernel_param = params->p_operation;
    float* p_act_data = params->p_act_data;
    half* p_act_out = params->p_act_out;
    call_counter++;
    
    const uint32_t elements
        = p_kernel_param->iw * p_kernel_param->ih * p_kernel_param->ic;
    float act = 0;
    
    for (uint32_t e = 0; e < elements; ++e) {
        act = *p_act_data++ * -1.0f;
        act = 1.0f + expf(act);
        act = 1.0f / act;
        // activation
        *p_act_out++ = (half)act;
    }
}