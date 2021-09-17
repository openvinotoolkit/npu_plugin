// {% copyright %}

#pragma once

#include <sw_layer_params.h>

#include <mv_types.h>

#ifdef __MOVICOMPILE__
#    include <moviVectorTypes.h>
#else
typedef fp16 half;
#endif

namespace nn {
namespace shave_lib {

struct t_MvSoftMaxParamNClasses : public LayerParams
{
    const half* input;
    half* output;
    u8* cmxslice;
    u8* auxcmxslice;
    u32 grpLeaderShave;
    u32 grpShavesNo;

    s32 ndims;
    s32 in_dims[MAX_ND_DIMS];
    s32 in_strides[MAX_ND_DIMS];
    s32 out_strides[MAX_ND_DIMS];

     s32 axis;
    s32 axisDim;
    s32 axisIStride;
    s32 axisOStride;
    s32 start;
    s32 toProcess;
    s32 this_shave;
};

} // namespace shave_lib
} // namespace nn
