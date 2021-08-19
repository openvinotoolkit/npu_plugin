// {% copyright %}
#pragma once

#include "sw_layer_params.h"
#include <mv_types.h>

#ifdef __MOVICOMPILE__
#    include <moviVectorTypes.h>
#else
typedef fp16 half;
#endif

namespace nn {
namespace shave_lib {

typedef struct __attribute__((packed))
{
    float min;
    float max;
} t_ClampLayerParams;

typedef struct __attribute__((packed))
{
    float shift;
    float scale;
    float power;
} t_PowerLayerParams;

typedef struct __attribute__((packed))
{
    float beta;
} t_SwishLayerParams;

enum class roundMode {
    HALF_TO_EVEN = 0,
    HALF_AWAY_FROM_ZERO = 1
};

typedef struct __attribute__((packed))
{
    roundMode mode;
} t_RoundLayerParams;

enum t_PostOps
{
    CLAMP,
    ELU,
    POWER,
    BIAS_LEAKY_RELU,
    BIAS_RELU,
    LEAKY_RELU,
    RELU,
    PRELU,
    SIGMOID,
    TANH,
    SCALE,
    BIAS,
    SCALE_SHIFT,
    HSWISH,
    SWISH,
    SOFTPLUS,
    MISH,
    FLOOR,
    CEIL,
    ROUND,
    ERF,
    GELU,
    LOG,
    EXP
};

struct PostOpsParams : LayerParams {
    const half *input;
    half *output;
    const half *weights;
    const half *bias;
    NDOrder order;
    u32   width;
    u32   height;
    u32   channels;
    u32   in_step;
    u32   out_step;
    u8*   cmxslice;
    t_PostOps postOpType;
    void *params = nullptr;

    bool has_weights = false;
    bool has_biases = false;
};

struct t_CHWPostOps3DParams : PostOpsParams
{
    int *ret_status = nullptr;
};

struct t_HCWPostOps3DParams : PostOpsParams
{
    u32 start_line = 0;
    u32 lines = 0;
    u32 offset = 0;
};

struct t_HWCPostOps3DParams : PostOpsParams
{
};

} // namespace shave_lib
} // namespace nn
