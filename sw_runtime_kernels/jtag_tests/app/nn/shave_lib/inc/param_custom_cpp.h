//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

//#include <sw_shave_performance.h>
#include <sw_layer_params.h>
#ifdef CONFIG_TARGET_SOC_3720
#include <sw_nn_runtime_types_3600.h>
#else
#include <sw_nn_runtime_types_2490.h>
#endif

#include <mv_types.h>

#ifdef __MOVICOMPILE__
#   include <moviVectorTypes.h>
#else
    typedef fp16 half;
#endif

namespace nn {
namespace shave_lib {

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

typedef void (*Kernel)(uint32_t lParams);

struct alignas(64) CustomLayerCppParams : LayerParams {
    // Buffers etc for kernel, argument, sched info data
    // kernel code window pointer
    uint32_t kernelBuffer{ 0 };

    // address of the kernel
    uint32_t kernelOffset;

    // kernel arguments
    void *argBuffer { nullptr };
    sw_params::BaseKernelParams baseParamData;

    // size of the arguments array
    uint32_t argBufferSize;
    uint32_t localSecMemTotal;

    ScheduleInfo scheduleInfo{};
    uint64_t kernel = 0;
//    MvPerfStruct *perf{};
    bool moveToCmxIfNecessary = false;
};
#pragma pack(pop)

} // namespace shave_lib
} // namespace nn
