// {% copyright %}

#pragma once

#include <sw_layer_params.h>

#include <mv_types.h>

#ifdef __MOVICOMPILE__
#    include <moviVectorTypes.h>
#else
typedef fp16 half;
#endif

#include <common_types.h>

namespace sw_params {

struct __attribute__((packed)) SoftmaxParams {
    MemRefData input;
    MemRefData output;
    int32_t axis;
};

inline BaseKernelParams softmaxParamsToBaseKernelParams(SoftmaxParams * softmaxParams) {
    BaseKernelParams rezult;
    rezult.numInputs = 1;
    rezult.numOutputs = 1;
    rezult.inputsOffset = reinterpret_cast<uint8_t*>(&(softmaxParams->input)) - reinterpret_cast<uint8_t*>(softmaxParams);
    rezult.outputsOffset = reinterpret_cast<uint8_t*>(&(softmaxParams->output)) - reinterpret_cast<uint8_t*>(softmaxParams);
    return rezult;
}

} // sw_params
