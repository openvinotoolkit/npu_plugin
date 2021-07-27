#ifndef KERNEL_OP_H
#define KERNEL_OP_H
#include <stdint.h>

#ifdef _WIN32
#define ARCH_X86
#endif

#if __linux__
#define ARCH_X86
#endif

#ifdef ARCH_X86
typedef float half;
#else
#include <moviVectorUtils.h>
#endif

typedef struct {
    uint16_t iw;
    uint16_t ih;
    uint16_t ic;
    uint16_t ow;
    uint16_t oh;
    uint16_t oc;
    uint16_t fw : 8;
    uint16_t fh : 8;
    uint16_t stride_w : 8;
    uint16_t stride_h : 8;
    float leak_relu_alpha;
}kernel_op;

typedef struct {
    kernel_op* p_operation;
    float* p_act_data;
    half* p_act_out;
}slf_kernel_params;

typedef struct {
    slf_kernel_params params;
    kernel_op operations;
}slf_kernel_asds;
#endif

