/*
 * {% copyright %}
 */
#ifndef DMA_SHAVE_NN_H_
#define DMA_SHAVE_NN_H_

#include <cstdint>
#define uncached(x) (x)

#ifndef INLINE_ATTRIBUTE
# ifdef CONFIG_ALWAYS_INLINE
#  define INLINE_ATTRIBUTE inline __attribute((always_inline))
# else
#  define INLINE_ATTRIBUTE
# endif
#endif

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



    INLINE_ATTRIBUTE bool start(const void *a_src, void *a_dst, uint32_t byteLength);

    INLINE_ATTRIBUTE void wait();

    INLINE_ATTRIBUTE bool start(const void *a_src, void *a_dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                           uint32_t srcStride, uint32_t dstStride);

    INLINE_ATTRIBUTE bool start_pa(const void *src, void *dst, uint32_t byteLength);

    INLINE_ATTRIBUTE bool start_pa(const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth, uint32_t dstWidth,
                              uint32_t srcStride, uint32_t dstStride);

private:
    bool patch_va_;
    bool pending_;

    INLINE_ATTRIBUTE void patch(uint64_t &a) const;
};


#ifdef CONFIG_ALWAYS_INLINE
#include "../../3720/dma_shave_nn.cpp"
#endif

#endif  // DMA_SHAVE_NN_H_
