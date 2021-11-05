#include <math.h>
#include <moviVectorUtils.h>
#include <vau_exp_fp16_args.h>

void vau_exp_fp16(vau_exp_fp16_args *args) {
    half8 *p_act_data = (half8 *)(args->input + args->local_input_offset);
    half8 *p_act_out = (half8 *)(args->output + args->local_output_offset);
    const uint32_t elements = args->tensor_size / 8;

    for (uint32_t e = 0; e < elements; ++e) {
        *p_act_out++ = __builtin_shave_vau_exp_v8f16_r(*p_act_data++);
    }
}
