#pragma once

#include <cstdint>

#pragma pack(push, 1)

struct ScheduleInfo {
    uint32_t shaveId {};
    uint32_t nShaves {};
};

struct MemoryInfo {
    uint8_t *cmxData {nullptr};
    uint32_t availableCmxBytes {};
};

struct DmaAlShaveWrapper {
    void* dmaAlShaveHnd {nullptr};

    bool (*create)(void* dmaAlShaveHnd, const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                   uint32_t srcStride, uint32_t dstStride);
    bool (*add)(void* dmaAlShaveHnd, const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                uint32_t srcStride, uint32_t dstStride);
    bool (*start)(void* dmaAlShaveHnd, const void *a_src, void *a_dst, uint32_t byteLength);
    bool (*startStride)(void* dmaAlShaveHnd, const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                        uint32_t srcStride, uint32_t dstStride);

    bool (*create3D)(void* dmaAlShaveHnd, const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                     uint32_t srcStride, uint32_t dstStride, uint32_t numPlanes, uint32_t srcPlaneStride,
                     uint32_t dstPlaneStride);
    bool (*add3D)(void* dmaAlShaveHnd, const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                  uint32_t srcStride, uint32_t dstStride, uint32_t numPlanes, uint32_t srcPlaneStride,
                  uint32_t dstPlaneStride);
    void (*startTransfer)(void* dmaAlShaveHnd);
    bool (*start3D)(void* dmaAlShaveHnd, const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                    uint32_t srcStride, uint32_t dstStride, uint32_t numPlanes, uint32_t srcPlaneStride, uint32_t dstPlaneStride);

    void (*init4D)(void* dmaAlShaveHnd);
    void (*add4D)(void* dmaAlShaveHnd, const void* src, void* dst, uint32_t* ranges,
                  uint32_t* srcStrides, uint32_t* dstStrides, int64_t totalBytes, uint8_t dataTypeSize);
    void (*start4D)(void* dmaAlShaveHnd);

    void (*wait)(void* dmaAlShaveHnd);
};

struct KernelParams {
    ScheduleInfo scheduleInfo {};
    MemoryInfo memoryInfo {};
    DmaAlShaveWrapper dmaAlShaveWrapper {};
};

#pragma pack(pop)
