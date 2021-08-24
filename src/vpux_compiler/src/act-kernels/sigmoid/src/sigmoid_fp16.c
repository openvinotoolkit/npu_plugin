#include "../asds/dpu2p7_descriptor.h"
#include <math.h>
#include <moviVectorUtils.h>

#define USE __attribute__((used))
#define ARG_DATA __attribute__((section(".arg.data")))

ARG_DATA USE cfg_dpu_description descriptor_argument;

__attribute__((section(".uuid.rodata")))
const uint64_t uuid = 0x05063896D66C489D;


// This is how the activation SHAVE kernel will be called
// (kr->kernelEntry_)(ki->kernelArgs_.args_, ki->kernelArgs_.numArgs_);
void sigmoid_fp16(void* args, uint32_t num_args) {
    cfg_dpu_description* desc_ptr = (cfg_dpu_description*)args;
    uint32_t* debug_ptr = (uint32_t*)0x2e230000;
    *debug_ptr++ = (uint32_t)args;
    const uint32_t X = desc_ptr->idu.tensor_size0.x; // tensor_width
    const uint32_t Y = desc_ptr->idu.tensor_size0.y;
    const uint32_t Z = desc_ptr->idu.tensor_size1.z;
    uint32_t input_addr = desc_ptr->idu.tensor_start << 4;
    input_addr += desc_ptr->idu.act0_offset;
    *debug_ptr++ = input_addr;
    *debug_ptr++ = (uint32_t)desc_ptr->idu.tensor_start;
    *debug_ptr++ = (uint32_t)desc_ptr->idu.act0_offset;

    float* p_act_data = (float*)(input_addr); // 0x1F000000
    half* p_act_out = (half*)(desc_ptr->odu_ac_base.ac_base << 4); // 0x1F004000


    const uint32_t elements = X * Y * Z;
    float act = 0;

    for (uint32_t e = 0; e < elements; ++e) {
        act = *p_act_data++ * -1.0f;
        act = 1.0f + expf(act);
        act = 1.0f / act;
        *p_act_out++ = (half)act;
    }
}
