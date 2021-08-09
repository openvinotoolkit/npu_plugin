#include <stddef.h>
#include <stdint.h>
#include <moviVectorTypes.h>

typedef struct {
    half8 *src;
    half8 *dest;
    size_t elements;
} kernel_config;

__attribute__((section(".arg.data")))
__attribute__((used))
kernel_config config_data;

// FP16 tanh is new in MTL
void kernel_entry(void* args, uint32_t num_args) {
    kernel_config *cfg_ptr = (kernel_config*)args;
    half8 *src = cfg_ptr->src;
    half8 *dest = cfg_ptr->dest;
    size_t elements = cfg_ptr->elements;

    for (size_t i = 0; i < elements; ++i) {
        dest[i] = __builtin_shave_vau_tanh_r(src[i]);
    }
}

