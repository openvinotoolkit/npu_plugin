/*
 * {% copyright %}
 */
#pragma once

#include <cstdint>
//#include <ShDrvCmxDma.h>
#define uncached(x) (x)

class DmaAlShave
{
public:
    enum {
        // Note that actually the value of numPlanes-1 stored as U8 in transaction struct,
        // so values of 1..256 must be valid
        max_3D_planes = 256,
        // 8.5.5.1 / 8.5.5.2 Descriptor for 1D/2D Block Transfers
        // DMA job descriptor has limit for transaction length - 24 bits
        max_transfer_size = ((1 << 24) - 1)
    };

//    DmaAlShave();
//    ~DmaAlShave();


    inline __attribute((always_inline)) bool start(const void *a_src, void *a_dst, uint32_t byteLength) {
        memcpy_s(a_dst, byteLength, a_src, byteLength);
        return true;
    }

    inline __attribute((always_inline)) void wait() {
    }

    inline __attribute((always_inline)) bool start(const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                           uint32_t srcStride, uint32_t dstStride) {

        // Do not alter the state of the member variables.
        // This enables replaying the same transfer without
        // recreating it each time, just like the real DMA.

        const uint8_t *src = reinterpret_cast<const uint8_t *>(uncached(a_src));
        uint8_t *dst = reinterpret_cast<uint8_t *>(uncached(a_dst));

        for (uint32_t si = 0, di = 0, length = byteLength; length > 0;)
        {
            const uint32_t chunk = std::min(std::min(srcWidth - si, dstWidth - di), length);
            memcpy_s(dst, chunk, src, chunk);
            //        std::copy(src, src + chunk, dst);

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

    inline __attribute((always_inline)) bool start_pa(const void *src, void *dst, uint32_t byteLength) {
        patch_va_ = false;
        //    auto result = start(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), byteLength);
        auto result = start(src, dst, byteLength);
        patch_va_ = true;
        return result;
    }

    inline __attribute((always_inline)) bool start_pa(const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                              uint32_t srcStride, uint32_t dstStride) {
        patch_va_ = false;
        //    auto result = start(reinterpret_cast<uint32_t>(src), reinterpret_cast<uint32_t>(dst), byteLength,
        //        srcWidth, dstWidth, srcStride, dstStride);
        auto result = start(src, dst, byteLength,
                            srcWidth, dstWidth, srcStride, dstStride);
        patch_va_ = true;
        return result;
    }












//    bool start_pa(const void *a_src, void *a_dst, uint32_t byteLength);
//    bool start_pa(const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
//                  uint32_t srcStride, uint32_t dstStride);
//    bool start_pa(const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
//                  uint32_t srcStride, uint32_t dstStride, uint32_t numPlanes, uint32_t srcPlaneStride,
//                  uint32_t dstPlaneStride);
//
//    bool start(const void *a_src, void *a_dst, uint32_t byteLength);
//    bool start(const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
//               uint32_t srcStride, uint32_t dstStride);
//    bool start(const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
//               uint32_t srcStride, uint32_t dstStride, uint32_t numPlanes, uint32_t srcPlaneStride,
//               uint32_t dstPlaneStride);
//
//    bool start(uint64_t a_src, uint64_t a_dst, uint32_t byteLength);
//    bool start(uint64_t a_src, uint64_t a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
//               uint32_t srcStride, uint32_t dstStride);
//    bool start(uint64_t a_src, uint64_t a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
//               uint32_t srcStride, uint32_t dstStride, uint32_t numPlanes, uint32_t srcPlaneStride,
//               uint32_t dstPlaneStride);
//
//    bool start();
//    void wait();

private:
    //ShDrvCmxDmaTransaction transaction_;
    //ShDrvCmxDmaTransactionHnd handle_;
    bool patch_va_;
    bool pending_;

//    bool start_();

    inline __attribute((always_inline))  void patch(uint64_t &a) const
    {
#ifdef CONFIG_NN_L2C_PAGE_TABLE
        if (patch_va_)
            if (a >= 0x8000'0000ull)
                a |= static_cast<uint64_t>(CONFIG_NN_L2C_PAGE_TABLE) << 31;
#endif
    }
};

//class DmaAl4DShave
//{
//    enum
//    {
//        MAX_TRANSACTIONS = 16,
//    };
//
//public:
//    DmaAl4DShave();
//    ~DmaAl4DShave();
//
//    // Initialize 4D transfer
//    void init4D();
//
//    // Add a 4D transfer, but it is not running
//    void add4D(const void *src, void *dst, uint32_t *ranges, uint32_t *srcStrides, uint32_t *dstStrides,
//               int64_t totalBytes, uint8_t dataTypeSize);
//
//    // Add a 4D transfer, but it is not running
//    void add4D(uint64_t src, uint64_t dst, uint32_t *ranges, uint32_t *srcStrides, uint32_t *dstStrides,
//               int64_t totalBytes, uint8_t dataTypeSize);
//
//    // Start the added 4D transfers
//    void start4D();
//
//    void wait();
//
//private:
//    //ShDrvCmxDmaTransaction transactions_[MAX_TRANSACTIONS];
//    //ShDrvCmxDmaTransactionHnd handle_;
//    uint32_t count4D_;
//    bool pending_;
//
//    inline void patch(uint64_t &a) const
//    {
//#ifdef CONFIG_NN_L2C_PAGE_TABLE
//        if (a >= 0x8000'0000ull)
//            a |= static_cast<uint64_t>(CONFIG_NN_L2C_PAGE_TABLE) << 31;
//#endif
//    }
//};
