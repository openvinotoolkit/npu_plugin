/*
* {% copyright %}
*/
#pragma once

#include <cstdint>
#include <sw_layer_params.h>
#include <param_custom_cpp.h>

// Declaring PODs for the C ++ custom kernel API

#pragma pack(push, 1)
struct MemoryInfo {
    uint8_t *cmxData {nullptr};
    uint32_t availableCmxBytes {};

    void init(const nn::shave_lib::LayerParams& layerParams);
};

struct DmaAlShaveWrapper {
    void* dmaAlShaveHnd {nullptr};

    bool (*start)(DmaAlShaveWrapper* wrapper, const void *a_src, void *a_dst, uint32_t byteLength);
    bool (*startStride)(DmaAlShaveWrapper* wrapper, const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                  uint32_t srcStride, uint32_t dstStride);

    void (*startTransfer)(DmaAlShaveWrapper* wrapper);
    bool (*start3D)(DmaAlShaveWrapper* wrapper, const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                    uint32_t srcStride, uint32_t dstStride, uint32_t numPlanes, uint32_t srcPlaneStride, uint32_t dstPlaneStride);

    void (*wait)(DmaAlShaveWrapper* wrapper);

    void init(void* dmaHnd);
};

struct KernelParams {
    nn::shave_lib::ScheduleInfo scheduleInfo {};
    MemoryInfo memoryInfo {};
    DmaAlShaveWrapper dmaAlShaveWrapper {};
};

#pragma pack(pop)
