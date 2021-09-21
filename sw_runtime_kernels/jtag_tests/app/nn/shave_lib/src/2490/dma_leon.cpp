/*
* {% copyright %}
*/
#include "dma_leon.h"
#include <algorithm>

#ifdef DISPATCHER_BM
/* Bare Metal */
bool DmaAlLeon::start(const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                      uint32_t srcStride, uint32_t dstStride) {
    // const u8 *src = reinterpret_cast<const u8 *>(uncached(a_src));
    // u8 *dst = reinterpret_cast<u8 *>(uncached(a_dst));
    const u8 *src = reinterpret_cast<const u8 *>((a_src));
    u8 *dst = reinterpret_cast<u8 *>((a_dst));

    for (uint32_t si = 0, di = 0, length = byteLength; length > 0;) {
        const uint32_t chunk = std::min(std::min(srcWidth - si, dstWidth - di), length);
        // nnLog(MVLOG_DEBUG, " src: %p, dst: %p, chunk: %lu", src,dst,chunk);
        std::copy(src, src + chunk, dst);

        si += chunk;
        di += chunk;
        src += chunk;
        dst += chunk;
        length -= chunk;

        if (si == srcWidth) {
            si = 0;
            src += srcStride - srcWidth;
        }

        if (di == dstWidth) {
            di = 0;
            dst += dstStride - dstWidth;
        }
    }

    // TODO: check for return value
    return true;
}

void DmaAlLeon::wait(void) {
    // No op
}

#elif defined(DISPATCHER_OS)
/* RTEMS mode */

DmaAlLeon::DmaAlLeon() {
    nnLog(MVLOG_DEBUG, " [%d] %s DmaAlLeon Inst() created, OS mode", __LINE__, __func__);

    assert((((uint32_t)&handle_) & 0x3F) == 0 && "Handle not aligned!");
    assert((((uint32_t)&transaction_) & 0x3F) == 0 && "Transaction not aligned!");

    // Don't memset the handle_ here.
    memset(&transaction_, 0, sizeof(transaction_));
    memset(&handle_, 0, sizeof(handle_));
    int priority = 0; // TODO: Make it configurable later
    // TODO: Check for proper initialization of this
#    if defined(__leon_nn__)
    OsDrvCmxDmaSetupStruct setup = {
        { .cpu_index = 0, .irq_priority = static_cast<uint8_t>(priority), .irq_enable = 1 }, nullptr
    };
#    else
    OsDrvCmxDmaSetupStruct setup = { .cpu_index = 0, .irq_priority = static_cast<uint8_t>(priority), .irq_enable = 1 };
#    endif /* __leon_nn__  */

    const int result = OsDrvCmxDmaInitialize(&setup);
    if (result == OS_MYR_DRV_ERROR) {
        nnLog(MVLOG_ERROR, " [%d] %s OsDrvCmxDmaInitialize() error, status = %d", __LINE__, __func__, result);
    } else {
        nnLog(MVLOG_DEBUG, " [%d] %s OsDrvCmxDmaInitialize() success, status = %d", __LINE__, __func__, result);
    }
}

DmaAlLeon::~DmaAlLeon() {
    nnLog(MVLOG_DEBUG, " [%d] %s DmaAlLeon cleanup begin", __LINE__, __func__);
    // De-Allocate handle_ and transaction_
    // TODO: Automatically wait for DMA to finish on destruct?
    nnLog(MVLOG_DEBUG, " [%d] %s DmaAlLeon cleanup done", __LINE__, __func__);
}

bool DmaAlLeon::start(const void *src, void *dst, uint32_t byteLength) {
    // if (transaction_ != nullptr) {
    // 1. Create Stride Transaction
    int result = OsDrvCmxDmaCreateTransaction(&handle_, &transaction_, reinterpret_cast<u8 *>(const_cast<void *>(src)),
                                              reinterpret_cast<u8 *>(dst), byteLength);
    if (result == OS_MYR_DRV_SUCCESS) {
        nnLog(MVLOG_DEBUG, " [%d] %s OsDrvCmxDmaCreateStrideTransaction success", __LINE__, __func__);
    } else {
        nnLog(MVLOG_ERROR, " [%d] %s OsDrvCmxDmaCreateStrideTransaction failure, result = %d", __LINE__, __func__,
              result);
    }

    const rtems_id task_id = rtems_task_self();
    nnLog(MVLOG_DEBUG, " RTEM TASK ID: %ld", task_id);
#    ifdef NCE_USE_DIRECT_LINK_MODE
    result = OsDrvCmxDmaStartRealTimeLinkAgent(&handle_, 0, &DmaAlLeon::callback, reinterpret_cast<void *>(task_id));
#    else
    result = OsDrvCmxDmaStartRealTimeTransfer(&handle_, &DmaAlLeon::callback, reinterpret_cast<void *>(task_id));
#    endif

    if (result == OS_MYR_DRV_SUCCESS) {
        nnLog(MVLOG_DEBUG, " [%d] %s OsDrvCmxDmaStartRealTimeTransfer success", __LINE__, __func__);
    } else {
        nnLog(MVLOG_ERROR, " [%d] %s OsDrvCmxDmaStartRealTimeTransfer failure, result = %d", __LINE__, __func__,
              result);
    }

    return result == OS_MYR_DRV_SUCCESS;
    /*} else {
        nnLog(MVLOG_DEBUG, "DMA Helper not available");
        return false;
    }*/
}

bool DmaAlLeon::start(const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                      uint32_t srcStride, uint32_t dstStride) {
    // if (transaction_ != nullptr) {
    // 1. Create Stride Transaction
    int result = OsDrvCmxDmaCreateStrideTransaction(
        &handle_, &transaction_, reinterpret_cast<u8 *>(const_cast<void *>(src)), reinterpret_cast<u8 *>(dst), srcWidth,
        dstWidth, srcStride, dstStride, byteLength);
    if (result == OS_MYR_DRV_SUCCESS) {
        nnLog(MVLOG_DEBUG, " [%d] %s OsDrvCmxDmaCreateStrideTransaction success", __LINE__, __func__);
    } else {
        nnLog(MVLOG_ERROR, " [%d] %s OsDrvCmxDmaCreateStrideTransaction failure, result = %d", __LINE__, __func__,
              result);
    }

    const rtems_id task_id = rtems_task_self();
    nnLog(MVLOG_DEBUG, " RTEM TASK ID: %ld", task_id);
#    ifdef NCE_USE_DIRECT_LINK_MODE
    result = OsDrvCmxDmaStartRealTimeLinkAgent(&handle_, 0, &DmaAlLeon::callback, reinterpret_cast<void *>(task_id));
#    else
    result = OsDrvCmxDmaStartRealTimeTransfer(&handle_, &DmaAlLeon::callback, reinterpret_cast<void *>(task_id));
#    endif

    if (result == OS_MYR_DRV_SUCCESS) {
        nnLog(MVLOG_DEBUG, " [%d] %s OsDrvCmxDmaStartRealTimeTransfer success", __LINE__, __func__);
    } else {
        nnLog(MVLOG_ERROR, " [%d] %s OsDrvCmxDmaStartRealTimeTransfer failure, result = %d", __LINE__, __func__,
              result);
    }

    return result == OS_MYR_DRV_SUCCESS;
    /*} else {
        nnLog(MVLOG_DEBUG, "DMA Helper not available");
        return false;
    }*/
}

void DmaAlLeon::wait() {
    nnLog(MVLOG_DEBUG, " [%d] %s waiting on Dma to finish", __LINE__, __func__);
    rtems_event_set event;
    while (rtems_event_system_receive(OSCOMMON_SYS_EVENT_CMX_DMA, RTEMS_WAIT | RTEMS_EVENT_ALL, RTEMS_NO_TIMEOUT,
                                      &event) != RTEMS_SUCCESSFUL ||
           event != OSCOMMON_SYS_EVENT_CMX_DMA)
        ;
    nnLog(MVLOG_DEBUG, " [%d] %s waiting on Dma done", __LINE__, __func__);
}

OsDrvCmxDmaTransactionHnd *DmaAlLeon::callback(OsDrvCmxDmaTransactionHnd *, void *user_context) {
    const rtems_id task_id = reinterpret_cast<rtems_id>(user_context);
    rtems_event_system_send(task_id, OSCOMMON_SYS_EVENT_CMX_DMA);
    return nullptr;
}

#endif /* DISPATCHER_BM / DISPATCHER_OS */
