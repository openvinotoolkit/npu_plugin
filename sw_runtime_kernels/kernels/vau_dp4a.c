#include <math.h>
#include <moviVectorUtils.h>
#include <vau_dp4a_args.h>

void vau_dp4a(vau_dp4a_args *args) {
    // integer multiply & accumulate over 16-byte groups
    // split the input buffer in half for the two input buffers
    uint32_t local_input = args->input + args->local_input_offset;
    schar16 *p_data1 = (schar16 *)(local_input + 0);
    schar16 *p_data2 = (schar16 *)(local_input + (args->tensor_size / 2));

    int4 *p_out = (int4 *)(args->output + args->local_output_offset);
    const uint32_t elements = args->tensor_size / 16;
    if (elements > 0) {
        // zero the accumulator before MAC
        *p_out++ = __builtin_shave_vau_dp4az_v16i8_vacc0_rr(*p_data1++, *p_data2++);
    }
    for (uint32_t e = 1; e < elements; e++) {
        // retain previous accumulator value
        *p_out++ = __builtin_shave_vau_dp4a_v16i8_vacc0_rr(*p_data1++, *p_data2++);
    }
}
