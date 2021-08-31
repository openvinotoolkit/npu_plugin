/*
* {% copyright %}
*/
#include <algorithm>
#include <math.h>
#include <param_custom_cpp.h>
#include <sw_shave_lib_common.h>
#include <sw_shave_performance.h>

#include <dma_shave.h>
#include <dma_shave_custom.h>
#include <nn_log.h>

#include <ShDrvCmxDma.h>
#include <mvMacros.h>
#include <stdio.h>

using nn::shave_lib::CustomLayerCppParams;

#define UPA_MAX_SHAVES CONFIG_SOC_SHAVE_COUNT
extern "C" {

typedef void prebuiltFunction_t(u32*, const KernelParams& kernelParams);

static inline void swcSetShaveWindow(uint32_t shaveNumber, uint32_t windowNumber, uint32_t targetWindowBaseAddr) {
    const uint32_t SVU_SLICE_OFFSET = 0x10000;
    uint32_t windowRegAddr =
        SHAVE_0_BASE_ADR + (SVU_SLICE_OFFSET * shaveNumber) + SLC_TOP_OFFSET_WIN_A + (windowNumber * 4);
    SET_REG_WORD(windowRegAddr, targetWindowBaseAddr);
}

static inline uint32_t swcGetShaveWindow(uint32_t shaveNumber, uint32_t windowNumber) {
    const uint32_t SVU_SLICE_OFFSET = 0x10000;
    uint32_t windowRegAddr =
        SHAVE_0_BASE_ADR + (SVU_SLICE_OFFSET * shaveNumber) + SLC_TOP_OFFSET_WIN_A + (windowNumber * 4);
    return GET_REG_WORD_VAL(windowRegAddr);
}

void custom_cpp(nn::shave_lib::CustomLayerCppParams* param) {
    // Stack memory is absolutely addressed and allocated per SHAVE on separate slice
    // It can be passed to cpp generated code as is

    const uint32_t shaveID = cpuWhoAmI() - PROCESS_SHAVE0;

#ifdef ENABLE_CUSTOM_KERNEL_PERF_COUNTERS
    auto perf = PerformanceCounters{param->perf, shaveID};
#endif

    int32_t res = ShDrvCmxDmaInitialize(nullptr);
    // check if something went wrong during driver initialization
    if (MYR_DRV_SUCCESS != res && MYR_DRV_ALREADY_INITIALIZED != res)
        return;

    // CMX slice offset ptr
    uint8_t *cmx = param->cmxData;
    const auto allocate = [&](int bytes, int alignment = 16) -> uint8_t * {
        auto aligned = (uint8_t *)ALIGN_UP((u32)cmx, alignment);
        cmx = aligned + bytes;
        return aligned;
    };

    // Allocate CMX memory for argument lists
    uint32_t* args = (uint32_t *)allocate(param->argBufferSize, 64);
    uint32_t localBufferPtr = (uint32_t)allocate(param->localSecMemTotal + 64, 1024);

    if (cmx > param->cmxData + param->availableCmxBytes) {
        nnLog(MVLOG_ERROR, "[CUSTOM CPP] NOT_ENOUGH_CMX");
        return;
    }

    memcpy_s(args, param->argBufferSize, param->argBuffer, param->argBufferSize);

    // get windows startup addresses
    uint32_t windowC = swcGetShaveWindow(shaveID, 2);
    uint32_t windowD = swcGetShaveWindow(shaveID, 3);

    // Map 0x1E000000 window to custom layer code
    swcSetShaveWindow(shaveID, 2, (unsigned int)(param->kernelBuffer));
    // Map 0x1F000000 window to custom layer data, which doesn't need to be initialised.
    swcSetShaveWindow(shaveID, 3, localBufferPtr);

    DmaAlShave dmaTask;
    MemoryInfo memoryInfo;
    DmaAlShaveWrapper dmaAlShaveWrapper;

    dmaAlShaveWrapper.init(&dmaTask);
    memoryInfo.init(*param);

    KernelParams kernelParams {
        param->scheduleInfo, memoryInfo, dmaAlShaveWrapper
    };

    prebuiltFunction_t *aux = (prebuiltFunction_t *)(param->kernelOffset);

#ifdef ENABLE_CUSTOM_KERNEL_PERF_COUNTERS
    perf.measureBegin();
#endif // ENABLE_CUSTOM_KERNEL_PERF_COUNTERS

    (*aux)(args, kernelParams);

#ifdef ENABLE_CUSTOM_KERNEL_PERF_COUNTERS
    perf.measureEnd();
#endif // ENABLE_CUSTOM_KERNEL_PERF_COUNTERS

    // The 0x1F000000 window to custom layer data, which doesn't need to be initialized
    swcSetShaveWindow(shaveID, 3, windowD);
    // Right before calling custom layer we are ready to move 0x1E000000 window to custom layer code
    swcSetShaveWindow(shaveID, 2, windowC);
}
}
