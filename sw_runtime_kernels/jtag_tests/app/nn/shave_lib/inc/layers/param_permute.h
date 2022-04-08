/*
* {% copyright %}
*/
#pragma once

#include "sw_layer_params.h"
#include "sw_tensor_ref.h"
#include <mv_types.h>

namespace nn {
namespace shave_lib {


typedef struct __attribute__((packed))
{
    uint32_t dims_in[MAX_ND_DIMS];
    uint32_t strides_in[MAX_ND_DIMS];
    uint32_t dims_out[MAX_ND_DIMS];
    uint32_t strides_out[MAX_ND_DIMS];
    uint32_t slice_sizes[MAX_ND_DIMS-2];
    uint32_t ndims;
    bool transpose;
} PermForDMA;

typedef struct __attribute__((packed))
{
    PermForDMA *parsedPerm;

    u32 bpp;

    const uint8_t* input;
    uint8_t* output;
    u32 slice;
    u32 n_slices;
    u32 sliceDivider;
    uint8_t* cmxData;
    int cmxSize;
} t_PermParam;

typedef struct __attribute__((packed))
{
    int32_t batch0;
    int32_t batch1;

    int32_t height0;
    int32_t height1;

    int32_t width;

    int32_t stride_in_line;
    int32_t stride_in_batch;
    int32_t stride_out_line;
    int32_t stride_out_batch;

    const uint8_t* input;
    uint8_t* output;

    uint8_t* cmxData;
    int cmxSize;
} t_PermTransposeParam;

struct PermuteParams : LayerParams {
    PermForDMA *parsedPerm = nullptr;
    u32 bpp = 0;
    bool run_mv_transpose = false;
    int n_of_slices = 0;
    int maxInnerDims = 0;
};

struct PermuteParams1D : LayerParams {
    u32 bpp = 0;
    u32 inWidth = 0;
    u32 inWidthStride = 0;
    u32 outWidthStride = 0;
};

struct mvPermuteParams : LayerParams {
    // FIXME join two structures
    // also, only one of them is actually used at this point
    union {
        t_PermTransposeParam mvPermTransposeParam;
        t_PermParam mvPermParam;
    } mvPermuteUnion;
    bool run_mv_transpose;
    bool is_shave_enabled;
};

} // namespace shave_lib
} // namespace nn
