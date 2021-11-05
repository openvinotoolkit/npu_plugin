/*
 * {% copyright %}
 */
#ifndef VAU_SIGM_FP16_ARGS_H_
#define VAU_SIGM_FP16_ARGS_H_
#include <stdint.h>

typedef struct {
    uint32_t input;
    uint32_t output;

    uint32_t tensor_size;
    uint32_t tensor_size_x;
    uint32_t tensor_size_y;
    uint32_t tensor_size_z;

    uint32_t local_input_offset;
    uint32_t local_output_offset;

    uint32_t args_size;
} vau_sigm_fp16_args;

#endif
