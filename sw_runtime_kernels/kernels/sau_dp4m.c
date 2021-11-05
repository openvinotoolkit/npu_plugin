#include <math.h>
#include <moviVectorUtils.h>
#include <sau_dp4m_args.h>

void sau_dp4m(sau_dp4m_args *args) {
    // dp4m is the same as dp4 except the second input is unsigned
    uint32_t local_input = args->input + args->local_input_offset;
    schar4 *p_data1 = (schar4 *)(local_input + 0);
    uchar4 *p_data2 = (uchar4 *)(local_input + (args->tensor_size / 2));
    int *p_out = (int *)(args->output + args->local_output_offset);

    const uint32_t elements = args->tensor_size / 8;
    for (uint32_t e = 0; e < elements; e++) {
        *p_out++ = __builtin_shave_sau_dp4m_v4i8_rr(*p_data1++, *p_data2++);
    }
}
