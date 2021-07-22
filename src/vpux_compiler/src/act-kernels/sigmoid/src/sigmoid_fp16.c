#include <math.h>
#include <moviVectorUtils.h>
#include <nn_act_args.h>

// #define USE __attribute__((used))
// #define ARG_DATA __attribute__((section(".arg.data")))

// This is how the activation SHAVE kernel will be called
// (kr->kernelEntry_)(ki->kernelArgs_.args_, ki->kernelArgs_.numArgs_);
void sigmoid_fp16(act_kernel_args *args) {
    float *p_act_data = (float *)(args->input + args->global_input_offset); // 0x1F000000
    half *p_act_out =   (half *)(args->output + args->global_output_offset); // 0x1F004000

    // const uint32_t elements = X * Y * Z;
    const uint32_t elements = args->tensor_size;
    float act = 0;

    for (uint32_t e = 0; e < elements; ++e) {
        act = *p_act_data++ * -1.0f;
        act = 1.0f + expf(act);
        act = 1.0f / act;
        *p_act_out++ = (half)act;
    }
}
