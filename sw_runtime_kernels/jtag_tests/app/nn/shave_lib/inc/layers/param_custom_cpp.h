/*
* {% copyright %}
*/
#pragma once

#include <sw_layer_params.h>
#include <sw_shave_performance.h>

#include <mv_types.h>

#ifdef __MOVICOMPILE__
#   include <moviVectorTypes.h>
#else
    typedef fp16 half;
#endif

namespace nn {
namespace shave_lib {

enum CustomCppOpIDs {
    SOFTMAX = 101,
    PAD,
};

typedef struct {
    uint32_t kernelEntry;
    uint32_t argumentsSize;
    uint32_t sec_mem_total;
} KernelCppDescriptor;

#pragma pack(push, 1)
struct ScheduleInfo {
    uint32_t shaveId;
    uint32_t nShaves;
};

struct alignas(64) CustomLayerCppParams : LayerParams {
    // Buffers etc for kernel, argument, sched info data
    // kernel code window pointer
    uint32_t kernelBuffer{ 0 };

    // address of the kernel
    uint32_t kernelOffset;

    // kernel arguments
    void *argBuffer { nullptr };

    // size of the arguments array
    uint32_t argBufferSize;
    uint32_t localSecMemTotal;

    // number of inputs/outputs
    uint32_t inputsSize{0};
    uint32_t outputsSize{0};

    ScheduleInfo scheduleInfo{};
    MvPerfStruct *perf{};
};
#pragma pack(pop)

} // namespace shave_lib
} // namespace nn
