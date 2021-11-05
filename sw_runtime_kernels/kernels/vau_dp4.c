#include <math.h>
#include <moviVectorUtils.h>
#include <vau_dp4_args.h>

void vau_dp4(vau_dp4_args *args) {
    // integer multiply & accumulate over 16-byte groups where the first
    // input is signed and the second input is unsigned. We will split the
    // input in half: low half is input 1, high half is input 2.
    uint32_t local_input = args->input + args->local_input_offset;
    schar16 *p_data1 = (schar16 *)(local_input + 0);
    schar16 *p_data2 = (schar16 *)(local_input + (args->tensor_size / 2));

    int4 *p_out = (int4 *)(args->output + args->local_output_offset);
    const uint32_t elements = args->tensor_size / 16;
    for (uint32_t e = 0; e < elements; e++) {
        *p_out++ = __builtin_shave_vau_dp4_v16i8_rr(*p_data1++, *p_data2++);
    }
}
