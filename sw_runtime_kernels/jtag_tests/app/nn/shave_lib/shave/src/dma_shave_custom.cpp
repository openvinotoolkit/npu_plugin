/*
* {% copyright %}
*/
#include "dma_shave_custom.h"
#include "dma_shave.h"

namespace DmaAlShaveWrapperImpl {

bool start(DmaAlShaveWrapper* wrapper, const void *a_src, void *a_dst, uint32_t byteLength) {
    DmaAlShave* dmaAlShave = reinterpret_cast<DmaAlShave*>(wrapper->dmaAlShaveHnd);
    return dmaAlShave->start(a_src, a_dst, byteLength);
}

bool startStride(DmaAlShaveWrapper* wrapper, const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
           uint32_t srcStride, uint32_t dstStride) {
    DmaAlShave* dmaAlShave = reinterpret_cast<DmaAlShave*>(wrapper->dmaAlShaveHnd);
    return dmaAlShave->start(a_src, a_dst, byteLength, srcWidth, dstWidth, srcStride, dstStride);
}

void startTransfer(DmaAlShaveWrapper* wrapper) {
    DmaAlShave* dmaAlShave = reinterpret_cast<DmaAlShave*>(wrapper->dmaAlShaveHnd);
    dmaAlShave->start();
}

bool start3D(DmaAlShaveWrapper* wrapper, const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
             uint32_t srcStride, uint32_t dstStride, uint32_t numPlanes, uint32_t srcPlaneStride, uint32_t dstPlaneStride) {
    DmaAlShave* dmaAlShave = reinterpret_cast<DmaAlShave*>(wrapper->dmaAlShaveHnd);
    return dmaAlShave->start(a_src, a_dst, byteLength, srcWidth, dstWidth,
                             srcStride, dstStride, numPlanes, srcPlaneStride, dstPlaneStride);
}

void wait(DmaAlShaveWrapper* wrapper) {
    DmaAlShave* dmaAlShave = reinterpret_cast<DmaAlShave*>(wrapper->dmaAlShaveHnd);
    dmaAlShave->wait();
}

} // namespace DmaAlShaveWrapperImpl

void MemoryInfo::init(const nn::shave_lib::LayerParams& layerParams) {
    cmxData = layerParams.cmxData;
    availableCmxBytes = layerParams.availableCmxBytes;
}

void DmaAlShaveWrapper::init(void* dmaHnd) {
    dmaAlShaveHnd = dmaHnd;

    start = DmaAlShaveWrapperImpl::start;
    startStride = DmaAlShaveWrapperImpl::startStride;

    startTransfer = DmaAlShaveWrapperImpl::startTransfer;
    start3D = DmaAlShaveWrapperImpl::start3D;

    wait = DmaAlShaveWrapperImpl::wait;
}
