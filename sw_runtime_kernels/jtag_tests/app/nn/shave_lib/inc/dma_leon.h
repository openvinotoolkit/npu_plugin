/*
* {% copyright %}
*/
#pragma once

#include <cstdint>
#include <nn_log.h>

typedef uint32_t uint32_t;
typedef unsigned char u8;

#ifdef DISPATCHER_BM
class DmaAlLeon {
  private:
  public:
    DmaAlLeon() { nnLog(MVLOG_DEBUG, " [%d] %s DmaAlLeon Inst() created, BM mode \n", __LINE__, __func__); }

    bool start(const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
               uint32_t srcStride, uint32_t dstStride);

    void wait(void);

  public:
    enum
    {
        // Note that actually the value of numPlanes-1 stored as U8 in transaction struct,
        // so values of 1..256 must be valid
        max_3D_planes = 256,
        // 8.5.5.1 / 8.5.5.2 Descriptor for 1D/2D Block Transfers
        // DMA job descriptor has limit for transaction length - 24 bits
        max_transfer_size = ((1 << 24) - 1)
    };
};

#elif defined(DISPATCHER_OS)

#    include <OsDrvCmxDma.h>
#    include <rtems.h>

class alignas(64) DmaAlLeon {
  private:
    OsDrvCmxDmaTransactionHnd handle_ __attribute__((aligned(64)));
    OsDrvCmxDmaTransaction transaction_ __attribute__((aligned(64)));

    static OsDrvCmxDmaTransactionHnd *callback(OsDrvCmxDmaTransactionHnd *handle, void *user_context);

  public:
    DmaAlLeon();

    bool start(const void *src, void *dst, uint32_t byteLength);
    bool start(const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
               uint32_t srcStride, uint32_t dstStride);
    void wait(void);

    ~DmaAlLeon();

  public:
    enum
    {
        // Note that actually the value of numPlanes-1 stored as U8 in transaction struct,
        // so values of 1..256 must be valid
        max_3D_planes = 256,
        // 8.5.5.1 / 8.5.5.2 Descriptor for 1D/2D Block Transfers
        // DMA job descriptor has limit for transaction length - 24 bits
        max_transfer_size = ((1 << 24) - 1)
    };
};

#endif /* DISPATCHER_BM / DISPATCHER_OS */
