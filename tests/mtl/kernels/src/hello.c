#include <stddef.h>
#include <stdint.h>

typedef struct {
    float *src;
    float *dest;
    size_t elements;
} kernel_config;

// Reserve space in the volatile data for the kernel argument struct
// Do not refer to this variable directly; it will be provided as an argument
__attribute__((section(".arg.data")))
__attribute__((used))
kernel_config config_data;

// Entry point with standard activation-SHAVE kernel signature
void kernel_entry(void* args, uint32_t num_args) {
    kernel_config *cfg_ptr = (kernel_config*)args;
    float *src = cfg_ptr->src;
    float *dest = cfg_ptr->dest;
    size_t elements = cfg_ptr->elements;

    for (size_t i = 0; i < elements; ++i) {
        dest[i] = src[i] * 0.5f;
    }
}

