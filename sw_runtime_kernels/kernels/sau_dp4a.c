#include <math.h>
#include <moviVectorUtils.h>
#include <sau_dp4a_args.h>

void sau_dp4a(sau_dp4a_args *args) {
    // The test harness has no way of providing more than one input buffer,
    // so we will operate on adjacent pairs of schar4 input values.
    schar4 *p_data = (schar4 *)(args->input + args->local_input_offset);
    int *p_out = (int *)(args->output + args->local_output_offset);

    const uint32_t elements = args->tensor_size / 4;
    if (elements > 0) {
        schar4 left = *p_data++;
        schar4 right = *p_data++;
        *p_out++ = __builtin_shave_sau_dp4az_v4i8_sacc0_rr(left, right);
    }
    for (uint32_t e = 1; e < elements; e += 2) {
        schar4 left = *p_data++;
        schar4 right = *p_data++;
        *p_out++ = __builtin_shave_sau_dp4a_v4i8_sacc0_rr(left, right);
    }
}
