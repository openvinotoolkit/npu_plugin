/*
* {% copyright %}
*/
#include "dma_shave.h"
#include <string.h>
#include <nn_log.h>

DmaAlShave::DmaAlShave() :
    patch_va_(true),
    pending_(false)
{
    memset(&handle_, 0, sizeof(handle_));
    memset(&transaction_, 0, sizeof(transaction_));
}

DmaAlShave::~DmaAlShave()
{
    wait();
}

bool DmaAlShave::start_pa(const void *src, void *dst, uint32_t byteLength) {
    patch_va_ = false;
    auto result = start(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), byteLength);
    patch_va_ = true;
    return result;
}

bool DmaAlShave::start_pa(const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                          uint32_t srcStride, uint32_t dstStride) {
    patch_va_ = false;
    auto result = start(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), byteLength,
        srcWidth, dstWidth, srcStride, dstStride);
    patch_va_ = true;
    return result;
}

bool DmaAlShave::start_pa(const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                          uint32_t srcStride, uint32_t dstStride, uint32_t numPlanes, uint32_t srcPlaneStride,
                          uint32_t dstPlaneStride) {
    patch_va_ = false;
    auto result = start(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), byteLength,
        srcWidth, dstWidth, srcStride, dstStride, numPlanes, srcPlaneStride, dstPlaneStride);
    patch_va_ = true;
    return result;
}

bool DmaAlShave::start(const void *src, void *dst, uint32_t byteLength) {
    return start(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), byteLength);
}

bool DmaAlShave::start(const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                       uint32_t srcStride, uint32_t dstStride) {
    return start(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), byteLength,
        srcWidth, dstWidth, srcStride, dstStride);
}

bool DmaAlShave::start(const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                       uint32_t srcStride, uint32_t dstStride, uint32_t numPlanes, uint32_t srcPlaneStride,
                       uint32_t dstPlaneStride) {
    return start(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), byteLength,
        srcWidth, dstWidth, srcStride, dstStride, numPlanes, srcPlaneStride, dstPlaneStride);
}

bool DmaAlShave::start(uint64_t src, uint64_t dst, uint32_t byteLength) {
    patch(src);
    patch(dst);

    wait();
    memset(&transaction_, 0, sizeof(transaction_));

    int result = ShDrvCmxDmaCreateTransactionExt(
        &handle_, &transaction_, src, dst, byteLength);

    return result == MYR_DRV_SUCCESS ? start_() : false;
}

bool DmaAlShave::start(uint64_t src, uint64_t dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                       uint32_t srcStride, uint32_t dstStride) {
    patch(src);
    patch(dst);

    wait();
    memset(&transaction_, 0, sizeof(transaction_));

    int result = ShDrvCmxDmaCreateStrideTransactionExt(
        &handle_, &transaction_, src, dst,
        srcWidth, dstWidth, srcStride, dstStride, byteLength);

    return result == MYR_DRV_SUCCESS ? start_() : false;
}

bool DmaAlShave::start(uint64_t src, uint64_t dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                         uint32_t srcStride, uint32_t dstStride, uint32_t numPlanes, uint32_t srcPlaneStride,
                         uint32_t dstPlaneStride) {
    patch(src);
    patch(dst);

    wait();
    memset(&transaction_, 0, sizeof(transaction_));

    int result = ShDrvCmxDmaCreate3DTransactionExt(
        &handle_, &transaction_, src, dst,
        srcWidth, dstWidth, srcStride, dstStride,
        numPlanes, srcPlaneStride, dstPlaneStride, byteLength);

    return result == MYR_DRV_SUCCESS ? start_() : false;
}

bool DmaAlShave::start() {
    wait();
    return start_();
}

bool DmaAlShave::start_() {
    int result = ShDrvCmxDmaStartTransfer(&handle_);
    while (result == MYR_DRV_RESOURCE_BUSY)
        result = ShDrvCmxDmaStartTransfer(&handle_);

    if (result == MYR_DRV_SUCCESS)
        pending_ = true;
    else
        nnLog(MVLOG_ERROR, "DMA transaction failed to start");

    return result == MYR_DRV_SUCCESS;
}

void DmaAlShave::wait() {
    if (pending_)
        ShDrvCmxDmaWaitTransaction(&handle_);

    pending_ = false;
}

DmaAl4DShave::DmaAl4DShave() :
    count4D_(0),
    pending_(false)
{
    memset(&handle_, 0, sizeof(handle_));
    memset(&transactions_, 0, sizeof(transactions_));
}

DmaAl4DShave::~DmaAl4DShave()
{
}

// Initialize 4D transfer
void DmaAl4DShave::init4D()
{
    wait();
    count4D_ = 0;
}

// Add a 4D DMA transfer
void DmaAl4DShave::add4D(const void* src, void* dst, uint32_t* ranges,
    uint32_t* srcStrides, uint32_t* dstStrides, int64_t totalBytes, uint8_t dataTypeSize) {

    add4D(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), ranges,
        srcStrides, dstStrides, totalBytes, dataTypeSize);
}

void DmaAl4DShave::add4D(uint64_t src, uint64_t dst, uint32_t *ranges,
    uint32_t *srcStrides, uint32_t *dstStrides, int64_t totalBytes, uint8_t dataTypeSize) {

    patch(src);
    patch(dst);

    int result;
    if (ranges[0] > 1) {
#ifdef DMA_TRANSFER_4D
        // Real 4D, should be replaced with one API when available
        int64_t size3D = ranges[1] * ranges[2] * ranges[3];
        size3D *= dataTypeSize;
        for (unsigned  i = 0; i < ranges[0]; ++i) {
            if (txCount == 0) {
                result = ShDrvCmxDmaCreate3DTransactionExt(&handle_, &transactions_[count4D_], src, dst,
                    ranges[3] * dataTypeSize, ranges[3] * dataTypeSize,
                    srcStrides[2] * dataTypeSize, dstStrides[2] * dataTypeSize,
                    ranges[1], srcStrides[1] * dataTypeSize, dstStrides[1] * dataTypeSize, totalBytes);
            }
            else {
                result = ShDrvCmxDmaAdd3DTransactionExt(&handle_, &transactions_[count4D_], src, dst,
                    ranges[3] * dataTypeSize, ranges[3] * dataTypeSize,
                    srcStrides[2] * dataTypeSize, dstStrides[2] * dataTypeSize,
                    ranges[1], srcStrides[1] * dataTypeSize, dstStrides[1] * dataTypeSize, totalBytes);
            }
            if (result != MYR_DRV_SUCCESS) {
                nnLog(MVLOG_ERROR, "Add DMA transfer failed.\n");
            }
            src += size3D;
            dst += size3D;
            ++count4D;
        }
#else
        nnLog(MVLOG_ERROR, "Cannot handle real 4D transfer.\n");
#endif
        return;
    }
    if (ranges[3] == srcStrides[2] && srcStrides[2] == dstStrides[2]) {
        if (ranges[2] * ranges[3] == srcStrides[1] && srcStrides[1] == dstStrides[1]) {
            // Could be 1D
            if (count4D_ == 0) {
                result = ShDrvCmxDmaCreateTransactionExt(&handle_, &transactions_[count4D_],
                    src, dst, totalBytes);
            }
            else {
                result = ShDrvCmxDmaAddTransactionExt(&handle_, &transactions_[count4D_],
                    src, dst, totalBytes);
            }
        }
        else {
            // Could be 2D
            if (count4D_ == 0) {
                result = ShDrvCmxDmaCreateStrideTransactionExt(&handle_, &transactions_[count4D_],
                    src, dst,
                    ranges[3] * ranges[2] * dataTypeSize, ranges[3] * ranges[2] * dataTypeSize,
                    srcStrides[1] * dataTypeSize, dstStrides[1] * dataTypeSize, totalBytes);
            }
            else {
                result = ShDrvCmxDmaAddStrideTransactionExt(&handle_, &transactions_[count4D_],
                    src, dst,
                    ranges[3] * ranges[2] * dataTypeSize, ranges[3] * ranges[2] * dataTypeSize,
                    srcStrides[1] * dataTypeSize, dstStrides[1] * dataTypeSize, totalBytes);
            }
        }
    }
    else {
        // 3D
        if (count4D_ == 0) {
            result = ShDrvCmxDmaCreate3DTransactionExt(&handle_, &transactions_[count4D_],
                src, dst,
                ranges[3] * dataTypeSize, ranges[3] * dataTypeSize,
                srcStrides[2] * dataTypeSize, dstStrides[2] * dataTypeSize,
                ranges[1], srcStrides[1] * dataTypeSize, dstStrides[1] * dataTypeSize, totalBytes);
        }
        else {
            result = ShDrvCmxDmaAdd3DTransactionExt(&handle_, &transactions_[count4D_],
                src, dst,
                ranges[3] * dataTypeSize, ranges[3] * dataTypeSize,
                srcStrides[2] * dataTypeSize, dstStrides[2] * dataTypeSize,
                ranges[1], srcStrides[1] * dataTypeSize, dstStrides[1] * dataTypeSize, totalBytes);
        }
    }
    if (result != MYR_DRV_SUCCESS)
        nnLog(MVLOG_ERROR, "Add DMA transfer failed.\n");

    ++count4D_;
}

void DmaAl4DShave::start4D() {
    if (count4D_ == 0)
        return;

    int result = ShDrvCmxDmaStartTransfer(&handle_);
    while(result == MYR_DRV_RESOURCE_BUSY)
        result = ShDrvCmxDmaStartTransfer(&handle_);

    if (result == MYR_DRV_SUCCESS)
        pending_ = true;
    else
        nnLog(MVLOG_ERROR, "DMA transaction failed to start");
}

void DmaAl4DShave::wait() {
    if (pending_)
        ShDrvCmxDmaWaitTransaction(&handle_);

    pending_ = false;
}
