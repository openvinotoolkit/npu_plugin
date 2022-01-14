//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#include <dma_shave_nn.h>
#include <string.h>
#include <algorithm>
#if defined(__leon__) || defined(__leon_nn__)
#include <rtems/rtems/cache.h>
#endif

INLINE_ATTRIBUTE bool DmaAlShave::start(const void *a_src, void *a_dst, uint32_t byteLength) {
#if defined(__leon__) || defined(__leon_nn__)
    rtems_cache_invalidate_multiple_data_lines(a_src, byteLength);
#endif
    memcpy(a_dst, a_src, byteLength);
#if defined(__leon__) || defined(__leon_nn__)
    rtems_cache_flush_multiple_data_lines(a_dst, byteLength);
#endif
    return true;
}

INLINE_ATTRIBUTE void DmaAlShave::wait() {
}

INLINE_ATTRIBUTE bool DmaAlShave::start(const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
           uint32_t srcStride, uint32_t dstStride) {

    // Do not alter the state of the member variables.
    // This enables replaying the same transfer without
    // recreating it each time, just like the real DMA.

    const uint8_t *src = reinterpret_cast<const uint8_t *>(uncached(a_src));
    uint8_t *dst = reinterpret_cast<uint8_t *>(uncached(a_dst));

    for (uint32_t si = 0, di = 0, length = byteLength; length > 0;)
    {
        const uint32_t chunk = std::min(std::min(srcWidth - si, dstWidth - di), length);
#if defined(__leon__) || defined(__leon_nn__)
        rtems_cache_invalidate_multiple_data_lines(src, chunk);
#endif
        memcpy(dst, src, chunk);
#if defined(__leon__) || defined(__leon_nn__)
    rtems_cache_flush_multiple_data_lines(dst, chunk);
#endif

        si += chunk;
        di += chunk;
        src += chunk;
        dst += chunk;
        length -= chunk;

        if (si == srcWidth)
        {
            si = 0;
            src += srcStride - srcWidth;
        }

        if (di == dstWidth)
        {
            di = 0;
            dst += dstStride - dstWidth;
        }
    }


    return true;
}

//DmaAlShave::DmaAlShave() :
//    patch_va_(true),
//    pending_(false)
//{
//    memset(&handle_, 0, sizeof(handle_));
//    memset(&transaction_, 0, sizeof(transaction_));
//}
//
//DmaAlShave::~DmaAlShave()
//{
//    wait();
//}
//
INLINE_ATTRIBUTE bool DmaAlShave::start_pa(const void *src, void *dst, uint32_t byteLength) {
    patch_va_ = false;
//    auto result = start(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), byteLength);
    auto result = start(src, dst, byteLength);
    patch_va_ = true;
    return result;
}

INLINE_ATTRIBUTE bool DmaAlShave::start_pa(const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                          uint32_t srcStride, uint32_t dstStride) {
    patch_va_ = false;
//    auto result = start(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), byteLength,
//        srcWidth, dstWidth, srcStride, dstStride);
    auto result = start(src, dst, byteLength,
        srcWidth, dstWidth, srcStride, dstStride);
    patch_va_ = true;
    return result;
}

INLINE_ATTRIBUTE void DmaAlShave::patch(uint64_t &a) const
{
#ifdef CONFIG_NN_L2C_PAGE_TABLE
    if (patch_va_)
        if (a >= 0x8000'0000ull)
            a |= static_cast<uint64_t>(CONFIG_NN_L2C_PAGE_TABLE) << 31;
#endif
}

//bool DmaAlShave::start(const void *src, void *dst, uint32_t byteLength) {
//    return start(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), byteLength);
//}
//
//bool DmaAlShave::start(const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
//                       uint32_t srcStride, uint32_t dstStride) {
//    return start(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), byteLength,
//        srcWidth, dstWidth, srcStride, dstStride);
//}
//
//bool DmaAlShave::start(const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
//                       uint32_t srcStride, uint32_t dstStride, uint32_t numPlanes, uint32_t srcPlaneStride,
//                       uint32_t dstPlaneStride) {
//    return start(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), byteLength,
//        srcWidth, dstWidth, srcStride, dstStride, numPlanes, srcPlaneStride, dstPlaneStride);
//}
//
//bool DmaAlShave::start(uint64_t src, uint64_t dst, uint32_t byteLength) {
//    patch(src);
//    patch(dst);
//
//    wait();
//    memset(&transaction_, 0, sizeof(transaction_));
//
//    int result = ShDrvCmxDmaCreateTransactionExt(
//        &handle_, &transaction_, src, dst, byteLength);
//
//    return result == MYR_DRV_SUCCESS ? start_() : false;
//}
//
//bool DmaAlShave::start(uint64_t src, uint64_t dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
//                       uint32_t srcStride, uint32_t dstStride) {
//    patch(src);
//    patch(dst);
//
//    wait();
//    memset(&transaction_, 0, sizeof(transaction_));
//
//    int result = ShDrvCmxDmaCreateStrideTransactionExt(
//        &handle_, &transaction_, src, dst,
//        srcWidth, dstWidth, srcStride, dstStride, byteLength);
//
//    return result == MYR_DRV_SUCCESS ? start_() : false;
//}
//
//bool DmaAlShave::start(uint64_t src, uint64_t dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
//                         uint32_t srcStride, uint32_t dstStride, uint32_t numPlanes, uint32_t srcPlaneStride,
//                         uint32_t dstPlaneStride) {
//    patch(src);
//    patch(dst);
//
//    wait();
//    memset(&transaction_, 0, sizeof(transaction_));
//
//    int result = ShDrvCmxDmaCreate3DTransactionExt(
//        &handle_, &transaction_, src, dst,
//        srcWidth, dstWidth, srcStride, dstStride,
//        numPlanes, srcPlaneStride, dstPlaneStride, byteLength);
//
//    return result == MYR_DRV_SUCCESS ? start_() : false;
//}
//
//bool DmaAlShave::start() {
//    wait();
//    return start_();
//}
//
//bool DmaAlShave::start_() {
//    int result = ShDrvCmxDmaStartTransfer(&handle_);
//    while (result == MYR_DRV_RESOURCE_BUSY)
//        result = ShDrvCmxDmaStartTransfer(&handle_);
//
//    if (result == MYR_DRV_SUCCESS)
//        pending_ = true;
//    else
//        nnLog(MVLOG_ERROR, "DMA transaction failed to start");
//
//    return result == MYR_DRV_SUCCESS;
//}
//
//void DmaAlShave::wait() {
//    if (pending_)
//        ShDrvCmxDmaWaitTransaction(&handle_);
//
//    pending_ = false;
//}
//
//DmaAl4DShave::DmaAl4DShave() :
//    count4D_(0),
//    pending_(false)
//{
//    memset(&handle_, 0, sizeof(handle_));
//    memset(&transactions_, 0, sizeof(transactions_));
//}
//
//DmaAl4DShave::~DmaAl4DShave()
//{
//}
//
//// Initialize 4D transfer
//void DmaAl4DShave::init4D()
//{
//    wait();
//    count4D_ = 0;
//}
//
//// Add a 4D DMA transfer
//void DmaAl4DShave::add4D(const void* src, void* dst, uint32_t* ranges,
//    uint32_t* srcStrides, uint32_t* dstStrides, int64_t totalBytes, uint8_t dataTypeSize) {
//
//    add4D(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), ranges,
//        srcStrides, dstStrides, totalBytes, dataTypeSize);
//}
//
//void DmaAl4DShave::add4D(uint64_t src, uint64_t dst, uint32_t *ranges,
//    uint32_t *srcStrides, uint32_t *dstStrides, int64_t totalBytes, uint8_t dataTypeSize) {
//
//    patch(src);
//    patch(dst);
//
//    int result;
//    if (ranges[0] > 1) {
//#ifdef DMA_TRANSFER_4D
//        // Real 4D, should be replaced with one API when available
//        int64_t size3D = ranges[1] * ranges[2] * ranges[3];
//        size3D *= dataTypeSize;
//        for (unsigned  i = 0; i < ranges[0]; ++i) {
//            if (txCount == 0) {
//                result = ShDrvCmxDmaCreate3DTransactionExt(&handle_, &transactions_[count4D_], src, dst,
//                    ranges[3] * dataTypeSize, ranges[3] * dataTypeSize,
//                    srcStrides[2] * dataTypeSize, dstStrides[2] * dataTypeSize,
//                    ranges[1], srcStrides[1] * dataTypeSize, dstStrides[1] * dataTypeSize, totalBytes);
//            }
//            else {
//                result = ShDrvCmxDmaAdd3DTransactionExt(&handle_, &transactions_[count4D_], src, dst,
//                    ranges[3] * dataTypeSize, ranges[3] * dataTypeSize,
//                    srcStrides[2] * dataTypeSize, dstStrides[2] * dataTypeSize,
//                    ranges[1], srcStrides[1] * dataTypeSize, dstStrides[1] * dataTypeSize, totalBytes);
//            }
//            if (result != MYR_DRV_SUCCESS) {
//                nnLog(MVLOG_ERROR, "Add DMA transfer failed.\n");
//            }
//            src += size3D;
//            dst += size3D;
//            ++count4D;
//        }
//#else
//        nnLog(MVLOG_ERROR, "Cannot handle real 4D transfer.\n");
//#endif
//        return;
//    }
//    if (ranges[3] == srcStrides[2] && srcStrides[2] == dstStrides[2]) {
//        if (ranges[2] * ranges[3] == srcStrides[1] && srcStrides[1] == dstStrides[1]) {
//            // Could be 1D
//            if (count4D_ == 0) {
//                result = ShDrvCmxDmaCreateTransactionExt(&handle_, &transactions_[count4D_],
//                    src, dst, totalBytes);
//            }
//            else {
//                result = ShDrvCmxDmaAddTransactionExt(&handle_, &transactions_[count4D_],
//                    src, dst, totalBytes);
//            }
//        }
//        else {
//            // Could be 2D
//            if (count4D_ == 0) {
//                result = ShDrvCmxDmaCreateStrideTransactionExt(&handle_, &transactions_[count4D_],
//                    src, dst,
//                    ranges[3] * ranges[2] * dataTypeSize, ranges[3] * ranges[2] * dataTypeSize,
//                    srcStrides[1] * dataTypeSize, dstStrides[1] * dataTypeSize, totalBytes);
//            }
//            else {
//                result = ShDrvCmxDmaAddStrideTransactionExt(&handle_, &transactions_[count4D_],
//                    src, dst,
//                    ranges[3] * ranges[2] * dataTypeSize, ranges[3] * ranges[2] * dataTypeSize,
//                    srcStrides[1] * dataTypeSize, dstStrides[1] * dataTypeSize, totalBytes);
//            }
//        }
//    }
//    else {
//        // 3D
//        if (count4D_ == 0) {
//            result = ShDrvCmxDmaCreate3DTransactionExt(&handle_, &transactions_[count4D_],
//                src, dst,
//                ranges[3] * dataTypeSize, ranges[3] * dataTypeSize,
//                srcStrides[2] * dataTypeSize, dstStrides[2] * dataTypeSize,
//                ranges[1], srcStrides[1] * dataTypeSize, dstStrides[1] * dataTypeSize, totalBytes);
//        }
//        else {
//            result = ShDrvCmxDmaAdd3DTransactionExt(&handle_, &transactions_[count4D_],
//                src, dst,
//                ranges[3] * dataTypeSize, ranges[3] * dataTypeSize,
//                srcStrides[2] * dataTypeSize, dstStrides[2] * dataTypeSize,
//                ranges[1], srcStrides[1] * dataTypeSize, dstStrides[1] * dataTypeSize, totalBytes);
//        }
//    }
//    if (result != MYR_DRV_SUCCESS)
//        nnLog(MVLOG_ERROR, "Add DMA transfer failed.\n");
//
//    ++count4D_;
//}
//
//void DmaAl4DShave::start4D() {
//    if (count4D_ == 0)
//        return;
//
//    int result = ShDrvCmxDmaStartTransfer(&handle_);
//    while(result == MYR_DRV_RESOURCE_BUSY)
//        result = ShDrvCmxDmaStartTransfer(&handle_);
//
//    if (result == MYR_DRV_SUCCESS)
//        pending_ = true;
//    else
//        nnLog(MVLOG_ERROR, "DMA transaction failed to start");
//}
//
//void DmaAl4DShave::wait() {
//    if (pending_)
//        ShDrvCmxDmaWaitTransaction(&handle_);
//
//    pending_ = false;
//}
