// {% copyright %}

#include <math.h>
#include "param_pooling.h"


using namespace nn::act_shave_lib;

#if 0
#include <svuCommonShave.h>
#include <sw_tensor_ref.h>
#include "averageV3.h"
#include "dma_shave.h"
#include "sw_shave_lib_common.h"

#include <stdint.h>
#include <nn_log.h>

#define MIN(_a, _b) (__builtin_shave_cmu_min_i32_rr_int((_a), (_b)))
#define MAX(_a, _b) (__builtin_shave_cmu_max_i32_rr_int((_a), (_b)))
#define ALIGN_TO_MULTIPLE(_size, _val) (DIVR((_val), (_size)) * (_size))

using namespace nn::shave_lib;

void HWC_mvAvgPoolMxN(nn::shave_lib::PoolParams *p);
void CHW_mvAvgPoolMxN(nn::shave_lib::PoolParams *p);
void HWC_mvAvgPool3x3_runner(nn::shave_lib::PoolParams *p);
void HWC_mvAvgPool3x3(nn::shave_lib::PoolParams *p, DmaAlShave *dmaTask_);

static void dmaStart(DmaAlShave &dma, const void *src, void *dst, uint32_t byteLength, uint32_t srcWidth,
                     uint32_t dstWidth, uint32_t srcStride, uint32_t dstStride) {
    if (srcWidth == srcStride)
        srcWidth = srcStride = byteLength;
    if (dstWidth == dstStride)
        dstWidth = dstStride = byteLength;
    dma.start(src, dst, byteLength, srcWidth, dstWidth, srcStride, dstStride);
}

static void calcScales(int inBeg, int inSize, int outBeg, int outSize, int kernel, int stride, int pad, half *scales) {
    for (int o = 0; o < outSize; ++o) {
        const int ib0 = (outBeg + o) * stride - pad - inBeg;
        const int ie0 = (outBeg + o) * stride - pad + (kernel - 1) - inBeg;
        const int ib = MAX(ib0, 0);
        const int ie = MIN(ie0, inSize - 1);

        scales[o] = (half)1 / (half)(ie - ib + 1);
    }
}

typedef void HWC_SumRowProc(const half *src, half *dst, int rowVectors, int inBegY, int inSizeY, int outBegY,
                            int outSizeY, const nn::shave_lib::PoolParams *p);
typedef void HWC_SumColProc(const half *src, half *dst, int tileSizeY, int numChannels, int inBegX, int inSizeX,
                            int outBegX, int outSizeX, int inRowSize, int outRowSize,
                            const nn::shave_lib::PoolParams *p);

static void HWC_sumRow_VK(const half *_src, half *_dst, int rowVectors, int inBegY, int inSizeY, int outBegY,
                          int outSizeY, const nn::shave_lib::PoolParams *p) {
    const half8 *src = (const half8 *)_src;
    half8 *dst = (half8 *)_dst;

    const int KY = p->radixY;
    const int SY = p->strideY;
    const int PY = p->padY;

    for (int oy = 0; oy < outSizeY; ++oy) {
        const int iby0 = (outBegY + oy) * SY - PY - inBegY;
        const int iey0 = (outBegY + oy) * SY - PY + (KY - 1) - inBegY;
        const int iby = MAX(iby0, 0);
        const int iey = MIN(iey0, inSizeY - 1);

        half8 *pdst = &dst[oy * rowVectors];

        int i = 0;
        for (; i < rowVectors - 7; i += 8) {
            const half8 *psrc = &src[iby * rowVectors + i];

            half8 hsum0 = psrc[0];
            half8 hsum1 = psrc[1];
            half8 hsum2 = psrc[2];
            half8 hsum3 = psrc[3];
            half8 hsum4 = psrc[4];
            half8 hsum5 = psrc[5];
            half8 hsum6 = psrc[6];
            half8 hsum7 = psrc[7];
            psrc += rowVectors;

            for (int iy = iby + 1; iy < iey + 1; ++iy) {
                hsum0 += psrc[0];
                hsum1 += psrc[1];
                hsum2 += psrc[2];
                hsum3 += psrc[3];
                hsum4 += psrc[4];
                hsum5 += psrc[5];
                hsum6 += psrc[6];
                hsum7 += psrc[7];
                psrc += rowVectors;
            }

            pdst[0] = hsum0;
            pdst[1] = hsum1;
            pdst[2] = hsum2;
            pdst[3] = hsum3;
            pdst[4] = hsum4;
            pdst[5] = hsum5;
            pdst[6] = hsum6;
            pdst[7] = hsum7;
            pdst += 8;
        }
        for (; i < rowVectors; ++i) {
            const half8 *psrc = &src[iby * rowVectors + i];

            half8 hsum = psrc[0];
            psrc += rowVectors;

            for (int iy = iby + 1; iy < iey + 1; ++iy) {
                hsum += psrc[0];
                psrc += rowVectors;
            }

            pdst[0] = hsum;
            ++pdst;
        }
    }
}

static void HWC_sumCol_YVK(const half *src, half *dst, int tileSizeY, int numChannels, int inBegX, int inSizeX,
                           int outBegX, int outSizeX, int inRowSize, int outRowSize,
                           const nn::shave_lib::PoolParams *p) {
    const int KX = p->radixX;
    const int SX = p->strideX;
    const int PX = p->padX;

    const int lastVector = numChannels % VECTOR_SIZE;

    const short8 seq = (short8){ 0, 1, 2, 3, 4, 5, 6, 7 };
    const short8 mask = (seq - (short8)lastVector) >> 15;

    for (int ox = 0; ox < outSizeX; ++ox) {
        const int ibx0 = (outBegX + ox) * SX - PX - inBegX;
        const int iex0 = (outBegX + ox) * SX - PX + (KX - 1) - inBegX;
        const int ibx = MAX(ibx0, 0);
        const int iex = MIN(iex0, inSizeX - 1);

        for (int ty = 0; ty < tileSizeY; ++ty) {
            const half *psrc = &src[ty * inRowSize + ibx * numChannels];
            half *pdst = (half *)&dst[ox * numChannels + ty * outRowSize];

            int i = 0;
            for (; i < numChannels - (VECTOR_SIZE * 4 - 1); i += VECTOR_SIZE * 4) {
                const half *p0 = &psrc[0 * VECTOR_SIZE];
                const half *p1 = &psrc[1 * VECTOR_SIZE];
                const half *p2 = &psrc[2 * VECTOR_SIZE];
                const half *p3 = &psrc[3 * VECTOR_SIZE];

                half8 hsum0 = *(const half8 *)p0;
                p0 += numChannels;
                half8 hsum1 = *(const half8 *)p1;
                p1 += numChannels;
                half8 hsum2 = *(const half8 *)p2;
                p2 += numChannels;
                half8 hsum3 = *(const half8 *)p3;
                p3 += numChannels;

                for (int ix = ibx + 1; ix < iex + 1; ++ix) {
                    hsum0 += *(const half8 *)p0;
                    p0 += numChannels;
                    hsum1 += *(const half8 *)p1;
                    p1 += numChannels;
                    hsum2 += *(const half8 *)p2;
                    p2 += numChannels;
                    hsum3 += *(const half8 *)p3;
                    p3 += numChannels;
                }

                *(half8 *)pdst = hsum0;
                pdst += VECTOR_SIZE;
                *(half8 *)pdst = hsum1;
                pdst += VECTOR_SIZE;
                *(half8 *)pdst = hsum2;
                pdst += VECTOR_SIZE;
                *(half8 *)pdst = hsum3;
                pdst += VECTOR_SIZE;

                psrc += VECTOR_SIZE * 4;
            }
            for (; i < numChannels - (VECTOR_SIZE - 1); i += VECTOR_SIZE) {
                const half *p = psrc;

                half8 hsum = *(const half8 *)p;
                p += numChannels;
                for (int ix = ibx + 1; ix < iex + 1; ++ix) {
                    hsum += *(const half8 *)p;
                    p += numChannels;
                }

                *(half8 *)pdst = hsum;
                pdst += VECTOR_SIZE;

                psrc += VECTOR_SIZE;
            }
            if (i < numChannels) {
                const half *p = psrc;

                half8 hsum = *(const half8 *)p;
                p += numChannels;
                for (int ix = ibx + 1; ix < iex + 1; ++ix) {
                    hsum += *(const half8 *)p;
                    p += numChannels;
                }

                *(half8 *)pdst = (half8)((*(short8 *)pdst & ~mask) | ((short8)hsum & mask));
            }
        }
    }
}

static void HWC_sumCol_VYK(const half *src, half *dst, int tileSizeY, int numChannels, int inBegX, int inSizeX,
                           int outBegX, int outSizeX, int inRowSize, int outRowSize,
                           const nn::shave_lib::PoolParams *p) {
    const int KX = p->radixX;
    const int SX = p->strideX;
    const int PX = p->padX;

    const int lastVector = numChannels % VECTOR_SIZE;

    const short8 seq = (short8){ 0, 1, 2, 3, 4, 5, 6, 7 };
    const short8 mask = (seq - (short8)lastVector) >> 15;

    for (int ox = 0; ox < outSizeX; ++ox) {
        const int ibx0 = (outBegX + ox) * SX - PX - inBegX;
        const int iex0 = (outBegX + ox) * SX - PX + (KX - 1) - inBegX;
        const int ibx = MAX(ibx0, 0);
        const int iex = MIN(iex0, inSizeX - 1);
        const int isx = iex - ibx + 1;

        if (isx == 1) {
            int i = 0;
            for (; i < numChannels - (VECTOR_SIZE - 1); i += VECTOR_SIZE) {
                const half *psrc = &src[i + ibx * numChannels];
                half *pdst = &dst[ox * numChannels + i];

                for (int ty = 0; ty < tileSizeY; ++ty) {
                    *(half8 *)pdst = *(const half8 *)psrc;
                    psrc += inRowSize;
                    pdst += outRowSize;
                }
            }
            if (i < numChannels) {
                const half *psrc = &src[i + ibx * numChannels];
                half *pdst = &dst[ox * numChannels + i];

                for (int ty = 0; ty < tileSizeY; ++ty) {
                    half8 hsum = *(const half8 *)psrc;

                    *(half8 *)pdst = (half8)((*(short8 *)pdst & ~mask) | ((short8)hsum & mask));

                    psrc += inRowSize;
                    pdst += outRowSize;
                }
            }
        } else if (isx == 2) {
            if (tileSizeY == 2) {
                int i = 0;
                for (; i < numChannels - (VECTOR_SIZE - 1); i += VECTOR_SIZE) {
                    const half *psrc = &src[i + ibx * numChannels];
                    half *pdst = &dst[ox * numChannels + i];

                    const half8 *p00 = (const half8 *)&psrc[0 * inRowSize + 0 * numChannels];
                    const half8 *p01 = (const half8 *)&psrc[0 * inRowSize + 1 * numChannels];
                    const half8 *p10 = (const half8 *)&psrc[1 * inRowSize + 0 * numChannels];
                    const half8 *p11 = (const half8 *)&psrc[1 * inRowSize + 1 * numChannels];

                    half8 hsum0 = p00[0] + p01[0];
                    half8 hsum1 = p10[0] + p11[0];

                    *(half8 *)pdst = hsum0;
                    pdst += outRowSize;
                    *(half8 *)pdst = hsum1;
                    pdst += outRowSize;
                }
                if (i < numChannels) {
                    const half *psrc = &src[i + ibx * numChannels];
                    half *pdst0 = &dst[ox * numChannels + 0 * outRowSize + i];
                    half *pdst1 = &dst[ox * numChannels + 1 * outRowSize + i];

                    const half8 *p00 = (const half8 *)&psrc[0 * inRowSize + 0 * numChannels];
                    const half8 *p01 = (const half8 *)&psrc[0 * inRowSize + 1 * numChannels];
                    const half8 *p10 = (const half8 *)&psrc[1 * inRowSize + 0 * numChannels];
                    const half8 *p11 = (const half8 *)&psrc[1 * inRowSize + 1 * numChannels];

                    half8 hsum0 = p00[0] + p01[0];
                    half8 hsum1 = p10[0] + p11[0];

                    *(half8 *)pdst0 = (half8)((*(short8 *)pdst0 & ~mask) | ((short8)hsum0 & mask));
                    *(half8 *)pdst1 = (half8)((*(short8 *)pdst1 & ~mask) | ((short8)hsum1 & mask));
                }
            } else {
                int i = 0;
                for (; i < numChannels - (VECTOR_SIZE - 1); i += VECTOR_SIZE) {
                    const half *psrc = &src[i + ibx * numChannels];
                    half *pdst = &dst[ox * numChannels + i];

                    int ty = 0;
                    for (; ty < tileSizeY - 3; ty += 4) {
                        const half8 *p00 = (const half8 *)&psrc[0 * inRowSize + 0 * numChannels];
                        const half8 *p01 = (const half8 *)&psrc[0 * inRowSize + 1 * numChannels];
                        const half8 *p10 = (const half8 *)&psrc[1 * inRowSize + 0 * numChannels];
                        const half8 *p11 = (const half8 *)&psrc[1 * inRowSize + 1 * numChannels];
                        const half8 *p20 = (const half8 *)&psrc[2 * inRowSize + 0 * numChannels];
                        const half8 *p21 = (const half8 *)&psrc[2 * inRowSize + 1 * numChannels];
                        const half8 *p30 = (const half8 *)&psrc[3 * inRowSize + 0 * numChannels];
                        const half8 *p31 = (const half8 *)&psrc[3 * inRowSize + 1 * numChannels];

                        half8 hsum0 = p00[0] + p01[0];
                        half8 hsum1 = p10[0] + p11[0];
                        half8 hsum2 = p20[0] + p21[0];
                        half8 hsum3 = p30[0] + p31[0];

                        *(half8 *)pdst = hsum0;
                        pdst += outRowSize;
                        *(half8 *)pdst = hsum1;
                        pdst += outRowSize;
                        *(half8 *)pdst = hsum2;
                        pdst += outRowSize;
                        *(half8 *)pdst = hsum3;
                        pdst += outRowSize;

                        psrc += inRowSize * 4;
                    }
                    for (; ty < tileSizeY; ++ty) {
                        const half8 *p0 = (const half8 *)&psrc[0 * numChannels];
                        const half8 *p1 = (const half8 *)&psrc[1 * numChannels];

                        half8 hsum = p0[0] + p1[0];

                        *(half8 *)pdst = hsum;
                        pdst += outRowSize;
                        psrc += inRowSize;
                    }
                }
                if (i < numChannels) {
                    const half *psrc = &src[i + ibx * numChannels];
                    half *pdst = &dst[ox * numChannels + i];

                    for (int ty = 0; ty < tileSizeY; ++ty) {
                        const half8 *p0 = (const half8 *)&psrc[0 * numChannels];
                        const half8 *p1 = (const half8 *)&psrc[1 * numChannels];

                        half8 hsum = p0[0] + p1[0];

                        *(half8 *)pdst = (half8)((*(short8 *)pdst & ~mask) | ((short8)hsum & mask));

                        psrc += inRowSize;
                        pdst += outRowSize;
                    }
                }
            }
        } else if (isx == 4) {
            int i = 0;
            for (; i < numChannels - (VECTOR_SIZE - 1); i += VECTOR_SIZE) {
                const half *psrc = &src[i + ibx * numChannels];
                half *pdst = &dst[ox * numChannels + i];

                int ty = 0;
                for (; ty < tileSizeY - 1; ty += 2) {
                    const half8 *p00 = (const half8 *)&psrc[0 * inRowSize + 0 * numChannels];
                    const half8 *p01 = (const half8 *)&psrc[0 * inRowSize + 1 * numChannels];
                    const half8 *p02 = (const half8 *)&psrc[0 * inRowSize + 2 * numChannels];
                    const half8 *p03 = (const half8 *)&psrc[0 * inRowSize + 3 * numChannels];
                    const half8 *p10 = (const half8 *)&psrc[1 * inRowSize + 0 * numChannels];
                    const half8 *p11 = (const half8 *)&psrc[1 * inRowSize + 1 * numChannels];
                    const half8 *p12 = (const half8 *)&psrc[1 * inRowSize + 2 * numChannels];
                    const half8 *p13 = (const half8 *)&psrc[1 * inRowSize + 3 * numChannels];

                    half8 hsum0 = p00[0] + p01[0] + p02[0] + p03[0];
                    half8 hsum1 = p10[0] + p11[0] + p12[0] + p13[0];

                    *(half8 *)pdst = hsum0;
                    pdst += outRowSize;
                    *(half8 *)pdst = hsum1;
                    pdst += outRowSize;

                    psrc += inRowSize * 2;
                }
                for (; ty < tileSizeY; ++ty) {
                    const half8 *p0 = (const half8 *)&psrc[0 * inRowSize + 0 * numChannels];
                    const half8 *p1 = (const half8 *)&psrc[0 * inRowSize + 1 * numChannels];
                    const half8 *p2 = (const half8 *)&psrc[0 * inRowSize + 2 * numChannels];
                    const half8 *p3 = (const half8 *)&psrc[0 * inRowSize + 3 * numChannels];

                    half8 hsum = p0[0] + p1[0] + p2[0] + p3[0];

                    *(half8 *)pdst = hsum;
                    pdst += outRowSize;

                    psrc += inRowSize;
                }
            }
            if (i < numChannels) {
                const half *psrc = &src[i + ibx * numChannels];
                half *pdst = &dst[ox * numChannels + i];

                int ty = 0;
                for (; ty < tileSizeY; ++ty) {
                    const half8 *p0 = (const half8 *)&psrc[0 * inRowSize + 0 * numChannels];
                    const half8 *p1 = (const half8 *)&psrc[0 * inRowSize + 1 * numChannels];
                    const half8 *p2 = (const half8 *)&psrc[0 * inRowSize + 2 * numChannels];
                    const half8 *p3 = (const half8 *)&psrc[0 * inRowSize + 3 * numChannels];

                    half8 hsum = p0[0] + p1[0] + p2[0] + p3[0];

                    *(half8 *)pdst = (half8)((*(short8 *)pdst & ~mask) | ((short8)hsum & mask));

                    psrc += inRowSize;
                    pdst += outRowSize;
                }
            }
        } else {
            int i = 0;
            for (; i < numChannels - (VECTOR_SIZE - 1); i += VECTOR_SIZE) {
                const half *psrc = &src[i + ibx * numChannels];
                half *pdst = &dst[ox * numChannels + i];

                int ty = 0;
                for (; ty < tileSizeY - 3; ty += 4) {
                    const half *p0 = &psrc[0 * inRowSize];
                    const half *p1 = &psrc[1 * inRowSize];
                    const half *p2 = &psrc[2 * inRowSize];
                    const half *p3 = &psrc[3 * inRowSize];

                    half8 hsum0 = *(const half8 *)p0;
                    p0 += numChannels;
                    half8 hsum1 = *(const half8 *)p1;
                    p1 += numChannels;
                    half8 hsum2 = *(const half8 *)p2;
                    p2 += numChannels;
                    half8 hsum3 = *(const half8 *)p3;
                    p3 += numChannels;

                    for (int ix = ibx + 1; ix < iex + 1; ++ix) {
                        hsum0 += *(const half8 *)p0;
                        p0 += numChannels;
                        hsum1 += *(const half8 *)p1;
                        p1 += numChannels;
                        hsum2 += *(const half8 *)p2;
                        p2 += numChannels;
                        hsum3 += *(const half8 *)p3;
                        p3 += numChannels;
                    }

                    *(half8 *)pdst = hsum0;
                    pdst += outRowSize;
                    *(half8 *)pdst = hsum1;
                    pdst += outRowSize;
                    *(half8 *)pdst = hsum2;
                    pdst += outRowSize;
                    *(half8 *)pdst = hsum3;
                    pdst += outRowSize;

                    psrc += inRowSize * 4;
                }
                for (; ty < tileSizeY; ++ty) {
                    const half *p = psrc;

                    half8 hsum = *(const half8 *)p;
                    p += numChannels;
                    for (int ix = ibx + 1; ix < iex + 1; ++ix) {
                        hsum += *(const half8 *)p;
                        p += numChannels;
                    }

                    *(half8 *)pdst = hsum;

                    psrc += inRowSize;
                    pdst += outRowSize;
                }
            }
            if (i < numChannels) {
                const half *psrc = &src[i + ibx * numChannels];

                int ty = 0;
                for (; ty < tileSizeY - 1; ty += 2) {
                    const half *p0 = &psrc[0 * inRowSize];
                    const half *p1 = &psrc[1 * inRowSize];

                    half *pdst0 = &dst[ox * numChannels + (ty + 0) * outRowSize + i];
                    half *pdst1 = &dst[ox * numChannels + (ty + 1) * outRowSize + i];

                    half8 hsum0 = *(const half8 *)p0;
                    p0 += numChannels;
                    half8 hsum1 = *(const half8 *)p1;
                    p1 += numChannels;

                    for (int ix = ibx + 1; ix < iex + 1; ++ix) {
                        hsum0 += *(const half8 *)p0;
                        p0 += numChannels;
                        hsum1 += *(const half8 *)p1;
                        p1 += numChannels;
                    }

                    *(half8 *)pdst0 = (half8)((*(short8 *)pdst0 & ~mask) | ((short8)hsum0 & mask));
                    *(half8 *)pdst1 = (half8)((*(short8 *)pdst1 & ~mask) | ((short8)hsum1 & mask));

                    psrc += inRowSize * 2;
                }
                for (; ty < tileSizeY; ++ty) {
                    const half *p = psrc;
                    half *pdst = &dst[ox * numChannels + ty * outRowSize + i];

                    half8 hsum = *(const half8 *)p;
                    p += numChannels;
                    for (int ix = ibx + 1; ix < iex + 1; ++ix) {
                        hsum += *(const half8 *)p;
                        p += numChannels;
                    }

                    *(half8 *)pdst = (half8)((*(short8 *)pdst & ~mask) | ((short8)hsum & mask));

                    psrc += inRowSize;
                }
            }
        }
    }
}

static void HWC_scaleZero(half *_dst, int outSizeY, int outRowSize, half8 areaScale) {
    half8 *dst = (half8 *)_dst;

    const int tileVectors = DIVR(outSizeY * outRowSize, VECTOR_SIZE);

    half8 *pdst = dst;

    int i = 0;

    if (tileVectors >= 6) {
        half8 r0 = pdst[0];
        half8 r1 = pdst[1];
        half8 r2 = pdst[2];
        half8 r3 = pdst[3];
        half8 r4 = pdst[4];
        half8 r5 = pdst[5];

        for (; i < tileVectors - 5; i += 6) {
            half8 r06 = pdst[6];
            half8 r07 = pdst[7];
            half8 r08 = pdst[8];
            half8 r09 = pdst[9];
            half8 r10 = pdst[10];
            half8 r11 = pdst[11];

            pdst[0] = r0 * areaScale;
            pdst[1] = r1 * areaScale;
            pdst[2] = r2 * areaScale;
            pdst[3] = r3 * areaScale;
            pdst[4] = r4 * areaScale;
            pdst[5] = r5 * areaScale;
            pdst += 6;

            r0 = r06;
            r1 = r07;
            r2 = r08;
            r3 = r09;
            r4 = r10;
            r5 = r11;
        }

        pdst[0] = r0 * areaScale;
        pdst[1] = r1 * areaScale;
        pdst[2] = r2 * areaScale;
        pdst[3] = r3 * areaScale;
        pdst[4] = r4 * areaScale;
        pdst[5] = r5 * areaScale;
        pdst += 6;
    }

    for (; i < tileVectors; ++i)
        *pdst++ *= areaScale;
}

static void HWC_scaleAvg(half *_dst, int outSizeY, int outSizeX, int numChannels, int outRowSize, const half *rowScales,
                         const half *colScales) {
    half *dst = (half *)_dst;

    for (int oy = 0; oy < outSizeY; ++oy) {
        half sy = (half)rowScales[oy];

        for (int ox = 0; ox < outSizeX; ++ox) {
            half sx = (half)colScales[ox];
            half scale = sy * sx;

            half *pdst = &dst[oy * outRowSize + ox * numChannels];

            int i = 0;
            for (; i < numChannels - (2 * VECTOR_SIZE - 1); i += 2 * VECTOR_SIZE) {
                *(half8 *)pdst *= (half8)scale;
                pdst += VECTOR_SIZE;
                *(half8 *)pdst *= (half8)scale;
                pdst += VECTOR_SIZE;
            }
            for (; i < numChannels; ++i) {
                *pdst *= scale;
                pdst++;
            }
        }
    }
}

void HWC_mvAvgPoolMxN(nn::shave_lib::PoolParams *p) {
    u8 *cmxslice = p->cmxslice;

    int const TC = p->channels;
    int const IH = p->inHeight;
    int const IW = p->inWidth;
    int const OH = p->outHeight;
    int const OW = p->outWidth;
    int const OS = p->outputStride / INPUT_BPP; // step
    int const KX = p->radixX;
    int const KY = p->radixY;
    int const SX = p->strideX;
    int const SY = p->strideY;
    int const PX = p->padX;
    int const PY = p->padY;

    int const widthTiles = p->widthTiles;
    int const channelTiles = p->channelTiles;
    int const inTileHeight = p->inTileHeight;
    int const inTileWidth = p->inTileWidth;
    int const outTileHeight = p->outTileHeight;
    int const outTileWidth = p->outTileWidth;
    int const tileChannels = p->tileChannels;

    int const firstTile = p->firstTile;
    int const numTiles = p->numTiles;

    int const tilingMode = p->tilingMode;
    int const rowsFirst = p->rowsFirst;
    int const fillInAvg = p->fillInAvg;

    if ((tilingMode != nn::shave_lib::TilingH) && (tilingMode != nn::shave_lib::TilingHW) &&
        (tilingMode != nn::shave_lib::TilingHC))
        return; // must be impossible due to LEON tiling assertion(s)

    DmaAlShave dma1;

    const int vectorAlign =
        (tilingMode == nn::shave_lib::TilingH) ? (MAX(VECTOR_SIZE, tileChannels) - tileChannels) : 0;
    const int inRowChannelsA = ALIGN_TO_MULTIPLE(VECTOR_SIZE, inTileWidth * tileChannels + vectorAlign);
    const int outRowChannelsA = ALIGN_TO_MULTIPLE(VECTOR_SIZE, outTileWidth * tileChannels + vectorAlign);

    const int inoutBufSize = MAX(inTileHeight * inRowChannelsA, outTileHeight * outRowChannelsA);
    const int tmpBufSize = rowsFirst ? (outTileHeight * inRowChannelsA) : (inTileHeight * outRowChannelsA);

    const int inoutBufSizeA = ALIGN_TO_MULTIPLE(CMX_ALIGN, inoutBufSize * INPUT_BPP) / INPUT_BPP;
    const int tmpBufSizeA = ALIGN_TO_MULTIPLE(CMX_ALIGN, tmpBufSize * INPUT_BPP) / INPUT_BPP;
    const int rowBufSizeA = ALIGN_TO_MULTIPLE(CMX_ALIGN, outTileHeight * INPUT_BPP) / INPUT_BPP;
    const int colBufSizeA = ALIGN_TO_MULTIPLE(CMX_ALIGN, outTileWidth * INPUT_BPP) / INPUT_BPP;

    for (uint32_t n = 0; n < p->batch_dim; ++n)
    {
        const fp16* input = p->input + (p->input_batch_step * n);
        fp16*  output           = p->output  + (p->output_batch_step * n);

        fp16 *cmx = (fp16 *)ALIGN_TO_MULTIPLE(CMX_ALIGN, (uint32_t)cmxslice);
        fp16 *inoutBuffer = cmx;
        cmx += inoutBufSizeA;
        fp16 *tmpBuffer = cmx;
        cmx += tmpBufSizeA;
        fp16 *rowScales = fillInAvg ? cmx : 0;
        cmx += fillInAvg * rowBufSizeA;
        fp16 *colScales = fillInAvg ? cmx : 0;
        cmx += fillInAvg * colBufSizeA;

        for (int tile = firstTile; tile < firstTile + numTiles; ++tile) {
            int outBegY = 0, outEndY = OH - 1;
            int outBegX = 0, outEndX = OW - 1;
            int begChannel = 0, numChannels = TC;

            int hTile = 0, wTile = 0, cTile = 0;
            switch (tilingMode) {
            case nn::shave_lib::TilingH:
                hTile = tile;
                outBegY = hTile * outTileHeight;
                outEndY = MIN(OH, outBegY + outTileHeight) - 1;
                break;
            case nn::shave_lib::TilingHW:
                hTile = tile / widthTiles;
                wTile = tile % widthTiles;
                outBegY = hTile * outTileHeight;
                outEndY = MIN(OH, outBegY + outTileHeight) - 1;
                outBegX = wTile * outTileWidth;
                outEndX = MIN(OW, outBegX + outTileWidth) - 1;
                break;
            case nn::shave_lib::TilingHC:
                hTile = tile / channelTiles;
                cTile = tile % channelTiles;
                outBegY = hTile * outTileHeight;
                outEndY = MIN(OH, outBegY + outTileHeight) - 1;
                begChannel = cTile * tileChannels;
                numChannels = MIN(TC - begChannel, tileChannels);
                break;
            }

            const int outSizeY = outEndY - outBegY + 1;
            const int outSizeX = outEndX - outBegX + 1;

            const int inBegY0 = outBegY * SY - PY;
            const int inEndY0 = outEndY * SY - PY + (KY - 1);
            const int inBegY = MAX(inBegY0, 0);
            const int inEndY = MIN(inEndY0, IH - 1);
            const int inSizeY = inEndY - inBegY + 1;

            const int inBegX0 = outBegX * SX - PX;
            const int inEndX0 = outEndX * SX - PX + (KX - 1);
            const int inBegX = MAX(inBegX0, 0);
            const int inEndX = MIN(inEndX0, IW - 1);
            const int inSizeX = (tilingMode == nn::shave_lib::TilingHW) ? (inEndX - inBegX + 1) : IW;

            const int tileSizeY = (rowsFirst ? outSizeY : inSizeY);

            const int inRowSize = inSizeX * numChannels;
            const int outRowSize = outSizeX * numChannels;
            const int inRowSizeA = ALIGN_TO_MULTIPLE(VECTOR_SIZE, inRowSize + vectorAlign);
            const int outRowSizeA = ALIGN_TO_MULTIPLE(VECTOR_SIZE, outRowSize + vectorAlign);
            const int numChannelsA = ALIGN_TO_MULTIPLE(VECTOR_SIZE, numChannels);

            switch (tilingMode) {
            case nn::shave_lib::TilingH:
                dmaStart(dma1,
                         input + ((inBegY * IW + inBegX) * TC + begChannel), // src
                         inoutBuffer,                                        // dst
                         inSizeY * inSizeX * numChannels * INPUT_BPP,        // byte length
                         inSizeY * inSizeX * numChannels * INPUT_BPP,        // src width
                         inRowSize * INPUT_BPP,                              // dst width
                         inSizeY * inSizeX * numChannels * INPUT_BPP,        // src stride
                         inRowSizeA * INPUT_BPP                              // dst stride
                );
                break;
            case nn::shave_lib::TilingHW:
                dmaStart(dma1,
                         input + ((inBegY * IW + inBegX) * TC + begChannel), // src
                         inoutBuffer,                                        // dst
                         inSizeY * inSizeX * numChannels * INPUT_BPP,        // byte length
                         inSizeX * numChannels * INPUT_BPP,                  // src width
                         inRowSize * INPUT_BPP,                              // dst width
                         IW * TC * INPUT_BPP,                                // src stride
                         inRowSizeA * INPUT_BPP                              // dst stride
                );
                break;
            case nn::shave_lib::TilingHC:
                dmaStart(dma1,
                         input + ((inBegY * IW + inBegX) * TC + begChannel), // src
                         inoutBuffer,                                        // dst
                         inSizeY * inSizeX * numChannels * INPUT_BPP,        // byte length
                         numChannels * INPUT_BPP,                            // src width
                         inRowSize * INPUT_BPP,                              // dst width
                         TC * INPUT_BPP,                                     // src stride
                         inRowSizeA * INPUT_BPP                              // dst stride
                );
                break;
            }
            dma1.wait();

            const int inRowVectors = inRowSizeA / VECTOR_SIZE;
            const int outRowVectors = outRowSizeA / VECTOR_SIZE;
            const int numVectors = numChannelsA / VECTOR_SIZE;

            HWC_SumRowProc *sumRow = HWC_sumRow_VK;
            HWC_SumColProc *sumCol = (numVectors < tileSizeY) ? HWC_sumCol_VYK : HWC_sumCol_YVK;

            if (rowsFirst) {
                sumRow(inoutBuffer, tmpBuffer, inRowVectors, inBegY, inSizeY, outBegY, outSizeY, p);
                sumCol(tmpBuffer, inoutBuffer, tileSizeY, numChannels, inBegX, inSizeX, outBegX, outSizeX, inRowSizeA,
                       outRowSizeA, p);
            } else {
                sumCol(inoutBuffer, tmpBuffer, tileSizeY, numChannels, inBegX, inSizeX, outBegX, outSizeX, inRowSizeA,
                       outRowSizeA, p);
                sumRow(tmpBuffer, inoutBuffer, outRowVectors, inBegY, inSizeY, outBegY, outSizeY, p);
            }

            if (fillInAvg) {
                calcScales(inBegY, inSizeY, outBegY, outSizeY, KY, SY, PY, rowScales);
                calcScales(inBegX, inSizeX, outBegX, outSizeX, KX, SX, PX, colScales);
                HWC_scaleAvg(inoutBuffer, outSizeY, outSizeX, numChannels, outRowSizeA, rowScales, colScales);
            } else {
                const half8 areaScale = (half8)(1.0f / float(KY * KX));
                HWC_scaleZero(inoutBuffer, outSizeY, outRowSizeA, areaScale);
            }

            switch (tilingMode) {
            case nn::shave_lib::TilingH:
                dmaStart(dma1,
                         inoutBuffer,                                           // src
                         output + ((outBegY * OW + outBegX) * OS + begChannel), // dst
                         outSizeY * outSizeX * numChannels * INPUT_BPP,         // byte length
                         outRowSize * INPUT_BPP,                                // src width
                         TC * INPUT_BPP,                                        // dst width
                         outRowSizeA * INPUT_BPP,                               // src stride
                         OS * INPUT_BPP                                         // dst stride
                );
                break;
            case nn::shave_lib::TilingHW:
                if (TC != OS) // with output stride
                {
                    for (int y = outBegY; y <= outEndY; ++y) {
                        dma1.wait();
                        dmaStart(dma1,
                                 inoutBuffer,                                     // src
                                 output + ((y * OW + outBegX) * OS + begChannel), // dst
                                 outSizeX * numChannels * INPUT_BPP,              // byte length
                                 numChannels * INPUT_BPP,                         // src width
                                 numChannels * INPUT_BPP,                         // dst width
                                 numChannels * INPUT_BPP,                         // src stride
                                 OS * INPUT_BPP                                   // dst stride
                        );
                    }
                } else // without output stride
                {
                    dmaStart(dma1,
                             inoutBuffer,                                           // src
                             output + ((outBegY * OW + outBegX) * OS + begChannel), // dst
                             outSizeY * outSizeX * numChannels * INPUT_BPP,         // byte length
                             outRowSize * INPUT_BPP,                                // src width
                             outRowSize * INPUT_BPP,                                // dst width
                             outRowSizeA * INPUT_BPP,                               // src stride
                             outRowSize * INPUT_BPP                                 // dst stride
                    );
                }
                break;
            case nn::shave_lib::TilingHC:
                dmaStart(dma1,
                         inoutBuffer,                                           // src
                         output + ((outBegY * OW + outBegX) * OS + begChannel), // dst
                         outSizeY * outSizeX * numChannels * INPUT_BPP,         // byte length
                         outRowSize * INPUT_BPP,                                // src width
                         numChannels * INPUT_BPP,                               // dst width
                         outRowSizeA * INPUT_BPP,                               // src stride
                         OS * INPUT_BPP                                         // dst stride
                );
                break;
            }
            dma1.wait();
        }
    }
}

typedef void CHW_SumRowProc(const half *src, half *dst, int numChannels, int rowVectors, int inBegY, int inSizeY,
                            int outBegY, int outSizeY, int inStepY, int outStepY,
                            const nn::shave_lib::PoolParams *p);
typedef void CHW_SumColProc(const half *src, half *dst, int numChannels, int tileSizeY, int inBegX, int outBegX,
                            int outSizeX, int inStepY, int outStepY, const nn::shave_lib::PoolParams *p);

static void CHW_fill(half *dst, int numLines, int lineStep, int fillSize) {
    const int numVectors = DIVR(fillSize, VECTOR_SIZE);

    for (int v = 0; v < numVectors; ++v) {
        half *pdst = &dst[v * VECTOR_SIZE];

#pragma clang loop unroll_count(5)
        for (int l = 0; l < numLines; ++l) {
            *(half8 *)pdst = (half8)0;
            pdst += lineStep;
        }
    }
}

static void CHW_sumRow(const half *src, half *dst, int numChannels, int rowVectors, int inBegY, int inSizeY,
                       int outBegY, int outSizeY, int inStepY, int outStepY, const nn::shave_lib::PoolParams *p) {
    const int KY = p->radixY;
    const int SY = p->strideY;
    const int PY = p->padY;

    const int rowSize = MIN(rowVectors * VECTOR_SIZE, outStepY);
    const int lastVector = rowSize % VECTOR_SIZE;

    const short8 seq = (short8){ 0, 1, 2, 3, 4, 5, 6, 7 };
    const short8 mask = (seq - (short8)lastVector) >> 15;

    for (int oy = 0; oy < outSizeY; ++oy) {
        const int iby0 = (outBegY + oy) * SY - PY - inBegY;
        const int iey0 = (outBegY + oy) * SY - PY + (KY - 1) - inBegY;
        const int iby = MAX(iby0, 0);
        const int iey = MIN(iey0, inSizeY - 1);
        const int isy = iey - iby + 1;

        int c = 0;
        for (; c < numChannels - 1; c += 2) {
            const half *psrc0 = &src[(c + 0) * inStepY * inSizeY + iby * inStepY];
            const half *psrc1 = &src[(c + 1) * inStepY * inSizeY + iby * inStepY];
            half *pdst0 = &dst[(c + 0) * outStepY * outSizeY + oy * outStepY];
            half *pdst1 = &dst[(c + 1) * outStepY * outSizeY + oy * outStepY];

            int i = 0;
            for (; i < rowSize - (4 * VECTOR_SIZE - 1); i += 4 * VECTOR_SIZE) {
                const half *ps0 = psrc0;
                const half *ps1 = psrc1;
                half *pd0 = pdst0;
                half *pd1 = pdst1;

                half8 hsum00 = *(const half8 *)(ps0 + 0 * VECTOR_SIZE);
                half8 hsum01 = *(const half8 *)(ps0 + 1 * VECTOR_SIZE);
                half8 hsum02 = *(const half8 *)(ps0 + 2 * VECTOR_SIZE);
                half8 hsum03 = *(const half8 *)(ps0 + 3 * VECTOR_SIZE);
                half8 hsum10 = *(const half8 *)(ps1 + 0 * VECTOR_SIZE);
                half8 hsum11 = *(const half8 *)(ps1 + 1 * VECTOR_SIZE);
                half8 hsum12 = *(const half8 *)(ps1 + 2 * VECTOR_SIZE);
                half8 hsum13 = *(const half8 *)(ps1 + 3 * VECTOR_SIZE);
                ps0 += inStepY;
                ps1 += inStepY;

                for (int iy = 1; iy < isy; ++iy) {
                    hsum00 += *(const half8 *)(ps0 + 0 * VECTOR_SIZE);
                    hsum01 += *(const half8 *)(ps0 + 1 * VECTOR_SIZE);
                    hsum02 += *(const half8 *)(ps0 + 2 * VECTOR_SIZE);
                    hsum03 += *(const half8 *)(ps0 + 3 * VECTOR_SIZE);
                    hsum10 += *(const half8 *)(ps1 + 0 * VECTOR_SIZE);
                    hsum11 += *(const half8 *)(ps1 + 1 * VECTOR_SIZE);
                    hsum12 += *(const half8 *)(ps1 + 2 * VECTOR_SIZE);
                    hsum13 += *(const half8 *)(ps1 + 3 * VECTOR_SIZE);
                    ps0 += inStepY;
                    ps1 += inStepY;
                }

                *(half8 *)(pd0 + 0 * VECTOR_SIZE) = hsum00;
                *(half8 *)(pd0 + 1 * VECTOR_SIZE) = hsum01;
                *(half8 *)(pd0 + 2 * VECTOR_SIZE) = hsum02;
                *(half8 *)(pd0 + 3 * VECTOR_SIZE) = hsum03;
                *(half8 *)(pd1 + 0 * VECTOR_SIZE) = hsum10;
                *(half8 *)(pd1 + 1 * VECTOR_SIZE) = hsum11;
                *(half8 *)(pd1 + 2 * VECTOR_SIZE) = hsum12;
                *(half8 *)(pd1 + 3 * VECTOR_SIZE) = hsum13;

                psrc0 += 4 * VECTOR_SIZE;
                psrc1 += 4 * VECTOR_SIZE;
                pdst0 += 4 * VECTOR_SIZE;
                pdst1 += 4 * VECTOR_SIZE;
            }
            for (; i < rowSize - (VECTOR_SIZE - 1); i += VECTOR_SIZE) {
                const half *ps0 = psrc0;
                const half *ps1 = psrc1;
                half *pd0 = pdst0;
                half *pd1 = pdst1;

                half8 hsum0 = *(const half8 *)ps0;
                ps0 += inStepY;
                half8 hsum1 = *(const half8 *)ps1;
                ps1 += inStepY;

                for (int iy = 1; iy < isy; ++iy) {
                    hsum0 += *(const half8 *)ps0;
                    ps0 += inStepY;
                    hsum1 += *(const half8 *)ps1;
                    ps1 += inStepY;
                }

                *(half8 *)pd0 = hsum0;
                *(half8 *)pd1 = hsum1;

                psrc0 += VECTOR_SIZE;
                psrc1 += VECTOR_SIZE;
                pdst0 += VECTOR_SIZE;
                pdst1 += VECTOR_SIZE;
            }
            if (i < rowSize) {
                const half *ps0 = psrc0;
                const half *ps1 = psrc1;
                half *pd0 = pdst0;
                half *pd1 = pdst1;

                half8 hsum0 = *(const half8 *)ps0;
                ps0 += inStepY;
                half8 hsum1 = *(const half8 *)ps1;
                ps1 += inStepY;

                for (int iy = 1; iy < isy; ++iy) {
                    hsum0 += *(const half8 *)ps0;
                    ps0 += inStepY;
                    hsum1 += *(const half8 *)ps1;
                    ps1 += inStepY;
                }

                *(half8 *)pd0 = (half8)((*(short8 *)pd0 & ~mask) | ((short8)hsum0 & mask));
                *(half8 *)pd1 = (half8)((*(short8 *)pd1 & ~mask) | ((short8)hsum1 & mask));
            }
        }
        for (; c < numChannels; ++c) {
            const half *psrc = &src[c * inStepY * inSizeY + iby * inStepY];
            half *pdst = &dst[c * outStepY * outSizeY + oy * outStepY];

            int i = 0;
            for (; i < rowSize - (4 * VECTOR_SIZE - 1); i += 4 * VECTOR_SIZE) {
                const half *ps = psrc;
                half *pd = pdst;

                half8 hsum0 = *(const half8 *)(ps + 0 * VECTOR_SIZE);
                half8 hsum1 = *(const half8 *)(ps + 1 * VECTOR_SIZE);
                half8 hsum2 = *(const half8 *)(ps + 2 * VECTOR_SIZE);
                half8 hsum3 = *(const half8 *)(ps + 3 * VECTOR_SIZE);
                ps += inStepY;

                for (int iy = 1; iy < isy; ++iy) {
                    hsum0 += *(const half8 *)(ps + 0 * VECTOR_SIZE);
                    hsum1 += *(const half8 *)(ps + 1 * VECTOR_SIZE);
                    hsum2 += *(const half8 *)(ps + 2 * VECTOR_SIZE);
                    hsum3 += *(const half8 *)(ps + 3 * VECTOR_SIZE);
                    ps += inStepY;
                }

                *(half8 *)(pd + 0 * VECTOR_SIZE) = hsum0;
                *(half8 *)(pd + 1 * VECTOR_SIZE) = hsum1;
                *(half8 *)(pd + 2 * VECTOR_SIZE) = hsum2;
                *(half8 *)(pd + 3 * VECTOR_SIZE) = hsum3;

                psrc += 4 * VECTOR_SIZE;
                pdst += 4 * VECTOR_SIZE;
            }
            for (; i < rowSize - (VECTOR_SIZE - 1); i += VECTOR_SIZE) {
                const half *ps = psrc;
                half *pd = pdst;

                half8 hsum = *(const half8 *)ps;
                ps += inStepY;
                for (int iy = 1; iy < isy; ++iy) {
                    hsum += *(const half8 *)ps;
                    ps += inStepY;
                }

                *(half8 *)pd = hsum;

                psrc += VECTOR_SIZE;
                pdst += VECTOR_SIZE;
            }
            if (i < rowSize) {
                const half *ps = psrc;
                half *pd = pdst;

                half8 hsum = *(const half8 *)ps;
                ps += inStepY;
                for (int iy = 1; iy < isy; ++iy) {
                    hsum += *(const half8 *)ps;
                    ps += inStepY;
                }

                *(half8 *)pd = (half8)((*(short8 *)pd & ~mask) | ((short8)hsum & mask));
            }
        }
    }
}

static void CHW_sumColN(const half *src, half *dst, int numChannels, int tileSizeY, int inBegX, int outBegX,
                        int outSizeX, int inStepY, int outStepY, const nn::shave_lib::PoolParams *p) {
    const int KX = p->radixX;
    const int SX = p->strideX;
    const int PX = p->padX;

    const int numVectors = DIVR(KX, VECTOR_SIZE);

    const short8 seq = (short8){ 0, 1, 2, 3, 4, 5, 6, 7 };
    const short8 mask = (seq - (short8)(KX - (numVectors - 1) * 8)) >> 15;

    for (int ox = 0; ox < outSizeX; ++ox) {
        const int ibx0 = (outBegX + ox) * SX - PX - inBegX;

        for (int ty = 0; ty < tileSizeY; ++ty) {
            int c = 0;
            for (; c < numChannels - 3; c += 4) {
                const half8 *psrc0 = (const half8 *)&src[((c + 0) * tileSizeY + ty) * inStepY + ibx0];
                const half8 *psrc1 = (const half8 *)&src[((c + 1) * tileSizeY + ty) * inStepY + ibx0];
                const half8 *psrc2 = (const half8 *)&src[((c + 2) * tileSizeY + ty) * inStepY + ibx0];
                const half8 *psrc3 = (const half8 *)&src[((c + 3) * tileSizeY + ty) * inStepY + ibx0];

                half *pdst0 = &dst[((c + 0) * tileSizeY + ty) * outStepY + ox];
                half *pdst1 = &dst[((c + 1) * tileSizeY + ty) * outStepY + ox];
                half *pdst2 = &dst[((c + 2) * tileSizeY + ty) * outStepY + ox];
                half *pdst3 = &dst[((c + 3) * tileSizeY + ty) * outStepY + ox];

                half8 hsum0 = psrc0[0];
                half8 hsum1 = psrc1[0];
                half8 hsum2 = psrc2[0];
                half8 hsum3 = psrc3[0];

                for (int i = 1; i < numVectors - 1; ++i) {
                    hsum0 += psrc0[i];
                    hsum1 += psrc1[i];
                    hsum2 += psrc2[i];
                    hsum3 += psrc3[i];
                }

                hsum0 += (half8)((short8)psrc0[numVectors - 1] & mask);
                hsum1 += (half8)((short8)psrc1[numVectors - 1] & mask);
                hsum2 += (half8)((short8)psrc2[numVectors - 1] & mask);
                hsum3 += (half8)((short8)psrc3[numVectors - 1] & mask);

                pdst0[0] = __builtin_shave_sau_sumx_f16_r(hsum0);
                pdst1[0] = __builtin_shave_sau_sumx_f16_r(hsum1);
                pdst2[0] = __builtin_shave_sau_sumx_f16_r(hsum2);
                pdst3[0] = __builtin_shave_sau_sumx_f16_r(hsum3);
            }
            for (; c < numChannels; ++c) {
                half *pdst = &dst[(c * tileSizeY + ty) * outStepY + ox];

                const half8 *psrc = (const half8 *)&src[(c * tileSizeY + ty) * inStepY + ibx0];

                half8 hsum = psrc[0];
                for (int i = 1; i < numVectors - 1; ++i)
                    hsum += psrc[i];
                hsum += (half8)((short8)psrc[numVectors - 1] & mask);

                half sum = __builtin_shave_sau_sumx_f16_r(hsum);

                pdst[0] = sum;
            }
        }
    }
}

static void CHW_sumCol1(const half *src, half *dst, int numChannels, int tileSizeY, int inBegX, int outBegX,
                        int outSizeX, int inStepY, int outStepY, const nn::shave_lib::PoolParams *p) {
    const int KX = p->radixX;
    const int SX = p->strideX;
    const int PX = p->padX;

    const short8 seq = (short8){ 0, 1, 2, 3, 4, 5, 6, 7 };
    const short8 mask = (seq - (short8)KX) >> 15;

    for (int c = 0; c < numChannels; ++c) {
        int ty = 0;
        for (; ty < tileSizeY - 1; ty += 2) {
            half *pdst0 = &dst[(c * tileSizeY + (ty + 0)) * outStepY + 0];
            half *pdst1 = &dst[(c * tileSizeY + (ty + 1)) * outStepY + 0];

            int ox = 0;
            for (; ox < outSizeX - 3; ox += 4) {
                const int ibx0 = (outBegX + ox) * SX - PX - inBegX;
                const half *psrc0 = &src[(c * tileSizeY + (ty + 0)) * inStepY + ibx0];
                const half *psrc1 = &src[(c * tileSizeY + (ty + 1)) * inStepY + ibx0];

                short8 wsrc00 = *(const short8 *)psrc0;
                psrc0 += SX;
                short8 wsrc01 = *(const short8 *)psrc0;
                psrc0 += SX;
                short8 wsrc02 = *(const short8 *)psrc0;
                psrc0 += SX;
                short8 wsrc03 = *(const short8 *)psrc0;
                psrc0 += SX;
                short8 wsrc10 = *(const short8 *)psrc1;
                psrc1 += SX;
                short8 wsrc11 = *(const short8 *)psrc1;
                psrc1 += SX;
                short8 wsrc12 = *(const short8 *)psrc1;
                psrc1 += SX;
                short8 wsrc13 = *(const short8 *)psrc1;
                psrc1 += SX;

                half8 hsum00 = (half8)(wsrc00 & mask);
                half8 hsum01 = (half8)(wsrc01 & mask);
                half8 hsum02 = (half8)(wsrc02 & mask);
                half8 hsum03 = (half8)(wsrc03 & mask);
                half8 hsum10 = (half8)(wsrc10 & mask);
                half8 hsum11 = (half8)(wsrc11 & mask);
                half8 hsum12 = (half8)(wsrc12 & mask);
                half8 hsum13 = (half8)(wsrc13 & mask);

                pdst0[ox + 0] = __builtin_shave_sau_sumx_f16_r(hsum00);
                pdst0[ox + 1] = __builtin_shave_sau_sumx_f16_r(hsum01);
                pdst0[ox + 2] = __builtin_shave_sau_sumx_f16_r(hsum02);
                pdst0[ox + 3] = __builtin_shave_sau_sumx_f16_r(hsum03);
                pdst1[ox + 0] = __builtin_shave_sau_sumx_f16_r(hsum10);
                pdst1[ox + 1] = __builtin_shave_sau_sumx_f16_r(hsum11);
                pdst1[ox + 2] = __builtin_shave_sau_sumx_f16_r(hsum12);
                pdst1[ox + 3] = __builtin_shave_sau_sumx_f16_r(hsum13);
            }
            for (; ox < outSizeX; ++ox) {
                const int ibx0 = (outBegX + ox) * SX - PX - inBegX;
                const half *psrc0 = &src[(c * tileSizeY + (ty + 0)) * inStepY + ibx0];
                const half *psrc1 = &src[(c * tileSizeY + (ty + 1)) * inStepY + ibx0];

                short8 wsrc0 = *(const short8 *)psrc0;
                short8 wsrc1 = *(const short8 *)psrc1;

                half8 hsum0 = (half8)(wsrc0 & mask);
                half8 hsum1 = (half8)(wsrc1 & mask);

                pdst0[ox] = __builtin_shave_sau_sumx_f16_r(hsum0);
                pdst1[ox] = __builtin_shave_sau_sumx_f16_r(hsum1);
            }
        }
        for (; ty < tileSizeY; ++ty) {
            half *pdst0 = &dst[(c * tileSizeY + (ty + 0)) * outStepY + 0];

            int ox = 0;
            for (; ox < outSizeX - 3; ox += 4) {
                const int ibx0 = (outBegX + ox) * SX - PX - inBegX;
                const half *psrc0 = &src[(c * tileSizeY + (ty + 0)) * inStepY + ibx0];

                short8 wsrc00 = *(const short8 *)psrc0;
                psrc0 += SX;
                short8 wsrc01 = *(const short8 *)psrc0;
                psrc0 += SX;
                short8 wsrc02 = *(const short8 *)psrc0;
                psrc0 += SX;
                short8 wsrc03 = *(const short8 *)psrc0;
                psrc0 += SX;

                half8 hsum00 = (half8)(wsrc00 & mask);
                half8 hsum01 = (half8)(wsrc01 & mask);
                half8 hsum02 = (half8)(wsrc02 & mask);
                half8 hsum03 = (half8)(wsrc03 & mask);

                pdst0[ox + 0] = __builtin_shave_sau_sumx_f16_r(hsum00);
                pdst0[ox + 1] = __builtin_shave_sau_sumx_f16_r(hsum01);
                pdst0[ox + 2] = __builtin_shave_sau_sumx_f16_r(hsum02);
                pdst0[ox + 3] = __builtin_shave_sau_sumx_f16_r(hsum03);
            }
            for (; ox < outSizeX; ++ox) {
                const int ibx0 = (outBegX + ox) * SX - PX - inBegX;
                const half *psrc0 = &src[(c * tileSizeY + (ty + 0)) * inStepY + ibx0];

                short8 wsrc0 = *(const short8 *)psrc0;

                half8 hsum0 = (half8)(wsrc0 & mask);

                pdst0[ox] = __builtin_shave_sau_sumx_f16_r(hsum0);
            }
        }
    }
}

static void CHW_scaleZero(half *_dst, int outSizeY, int outSizeX, int numChannels, half8 areaScale) {
    half8 *dst = (half8 *)_dst;

    const int numVectors = DIVR(outSizeY * outSizeX * numChannels, VECTOR_SIZE);

    half8 *pdst = dst;

    int i = 0;
    if (numVectors >= 6) {
        half8 r00 = pdst[0];
        half8 r01 = pdst[1];
        half8 r02 = pdst[2];
        half8 r03 = pdst[3];
        half8 r04 = pdst[4];
        half8 r05 = pdst[5];

        for (; i < numVectors - 5; i += 6) {
            half8 r06 = pdst[6];
            half8 r07 = pdst[7];
            half8 r08 = pdst[8];
            half8 r09 = pdst[9];
            half8 r10 = pdst[10];
            half8 r11 = pdst[11];

            pdst[0] = r00 * areaScale;
            pdst[1] = r01 * areaScale;
            pdst[2] = r02 * areaScale;
            pdst[3] = r03 * areaScale;
            pdst[4] = r04 * areaScale;
            pdst[5] = r05 * areaScale;
            pdst += 6;

            r00 = r06;
            r01 = r07;
            r02 = r08;
            r03 = r09;
            r04 = r10;
            r05 = r11;
        }

        pdst[0] = r00 * areaScale;
        pdst[1] = r01 * areaScale;
        pdst[2] = r02 * areaScale;
        pdst[3] = r03 * areaScale;
        pdst[4] = r04 * areaScale;
        pdst[5] = r05 * areaScale;
        pdst += 6;
    }

    for (; i < numVectors; ++i)
        *pdst++ *= areaScale;
}

static void CHW_scaleAvg(half *dst, int outSizeY, int outSizeX, int numChannels, const half *rowScales,
                         const half *colScales) {
    const int outPlaneStep = outSizeY * outSizeX;

    for (int oy = 0; oy < outSizeY; ++oy) {
        const half sy = (half)rowScales[oy];

        for (int ox = 0; ox < outSizeX; ++ox) {
            const half sx = (half)colScales[ox];
            const half scale = sy * sx;

            half *pdst = &dst[(0 * outSizeY + oy) * outSizeX + ox];

            int c = 0;
            if (numChannels >= 4) {
                half r0 = pdst[0 * outPlaneStep];
                half r1 = pdst[1 * outPlaneStep];
                half r2 = pdst[2 * outPlaneStep];
                half r3 = pdst[3 * outPlaneStep];

                for (; c < numChannels - 3; c += 4) {
                    half r4 = pdst[4 * outPlaneStep];
                    half r5 = pdst[5 * outPlaneStep];
                    half r6 = pdst[6 * outPlaneStep];
                    half r7 = pdst[7 * outPlaneStep];

                    pdst[0] = r0 * scale;
                    pdst += outPlaneStep;
                    pdst[0] = r1 * scale;
                    pdst += outPlaneStep;
                    pdst[0] = r2 * scale;
                    pdst += outPlaneStep;
                    pdst[0] = r3 * scale;
                    pdst += outPlaneStep;

                    r0 = r4;
                    r1 = r5;
                    r2 = r6;
                    r3 = r7;
                }

                pdst[0] = r0 * scale;
                pdst += outPlaneStep;
                pdst[0] = r1 * scale;
                pdst += outPlaneStep;
                pdst[0] = r2 * scale;
                pdst += outPlaneStep;
                pdst[0] = r3 * scale;
                pdst += outPlaneStep;
            }

            for (; c < numChannels; ++c) {
                pdst[0] *= scale;
                pdst += outPlaneStep;
            }
        }
    }
}

void CHW_mvAvgPoolMxN(nn::shave_lib::PoolParams *p) {
    u8 *cmxslice = p->cmxslice;

    int const TC = p->channels;
    int const IH = p->inHeight;
    int const IW = p->inWidth;
    int const OH = p->outHeight;
    int const OW = p->outWidth;
    int const IS = p->inputStride / INPUT_BPP;  // step
    int const OS = p->outputStride / INPUT_BPP; // step
    int const KX = p->radixX;
    int const KY = p->radixY;
    int const SX = p->strideX;
    int const SY = p->strideY;
    int const PX = p->padX;
    int const PY = p->padY;

    int const heightTiles = p->heightTiles;
    int const widthTiles = p->widthTiles;
    int const outTileHeight = p->outTileHeight;
    int const outTileWidth = p->outTileWidth;
    int const tileChannels = p->tileChannels;

    int const firstTile = p->firstTile;
    int const numTiles = p->numTiles;

    int const inPlaneSizeA = p->inPlaneSize;
    int const outPlaneSizeA = p->outPlaneSize;
    int const tmpPlaneSizeA = p->tmpPlaneSize;

    int const tilingMode = p->tilingMode;
    int const rowsFirst = p->rowsFirst;
    int const fillInAvg = p->fillInAvg;

    if ((tilingMode != nn::shave_lib::TilingC) && (tilingMode != nn::shave_lib::TilingCH) &&
        (tilingMode != nn::shave_lib::TilingCW))
        return; // must be impossible due to LEON tiling assertion(s)

    DmaAlShave dma1;

    const int inBufSize = inPlaneSizeA * tileChannels;
    const int outBufSize = outPlaneSizeA * tileChannels;
    const int tmpBufSize = tmpPlaneSizeA * tileChannels;

    const int inoutBufSize = MAX(inBufSize, outBufSize);

    const int inoutBufSizeA = ALIGN_TO_MULTIPLE(CMX_ALIGN, inoutBufSize * INPUT_BPP) / INPUT_BPP;
    const int tmpBufSizeA = ALIGN_TO_MULTIPLE(CMX_ALIGN, tmpBufSize * INPUT_BPP) / INPUT_BPP;
    const int rowBufSizeA = ALIGN_TO_MULTIPLE(CMX_ALIGN, outTileHeight * INPUT_BPP) / INPUT_BPP;
    const int colBufSizeA = ALIGN_TO_MULTIPLE(CMX_ALIGN, outTileWidth * INPUT_BPP) / INPUT_BPP;

    for (uint32_t n = 0; n < p->batch_dim; ++n)
    {
        const fp16* input = p->input + (p->input_batch_step * n);
        fp16*  output           = p->output  + (p->output_batch_step * n);

        fp16 *cmx = (fp16 *)ALIGN_TO_MULTIPLE(CMX_ALIGN, (uint32_t)cmxslice);
        fp16 *inoutBuffer = cmx;
        cmx += inoutBufSizeA;
        fp16 *tmpBuffer = cmx;
        cmx += tmpBufSizeA;
        fp16 *rowScales = fillInAvg ? cmx : 0;
        cmx += fillInAvg * rowBufSizeA;
        fp16 *colScales = fillInAvg ? cmx : 0;
        cmx += fillInAvg * colBufSizeA;

        CHW_SumRowProc *sumRow = CHW_sumRow;
        CHW_SumColProc *sumCol = (KX > 8) ? CHW_sumColN : CHW_sumCol1;

        for (int tile = firstTile; tile < firstTile + numTiles; ++tile) {
            int outBegY = 0, outEndY = OH - 1;
            int outBegX = 0, outEndX = OW - 1;
            int begChannel = 0, numChannels = TC;

            int cTile = 0, hTile = 0, wTile = 0;
            switch (tilingMode) {
            case nn::shave_lib::TilingC:
                cTile = tile;
                begChannel = cTile * tileChannels;
                numChannels = MIN(TC - begChannel, tileChannels);
                break;
            case nn::shave_lib::TilingCH:
                cTile = tile / heightTiles;
                hTile = tile % heightTiles;
                begChannel = cTile * tileChannels;
                numChannels = MIN(TC - begChannel, tileChannels);
                outBegY = hTile * outTileHeight;
                outEndY = MIN(OH, outBegY + outTileHeight) - 1;
                break;
            case nn::shave_lib::TilingCW:
                cTile = tile / widthTiles;
                wTile = tile % widthTiles;
                begChannel = cTile * tileChannels;
                numChannels = MIN(TC - begChannel, tileChannels);
                outBegX = wTile * outTileWidth;
                outEndX = MIN(OW, outBegX + outTileWidth) - 1;
                break;
            }

            const int outSizeY = outEndY - outBegY + 1;
            const int outSizeX = outEndX - outBegX + 1;

            const int inBegY0 = outBegY * SY - PY;
            const int inEndY0 = outEndY * SY - PY + (KY - 1);
            const int inBegY = MAX(inBegY0, 0);
            const int inEndY = MIN(inEndY0, IH - 1);
            const int inSizeY = inEndY - inBegY + 1;

            const int inBegX0 = outBegX * SX - PX;
            const int inEndX0 = outEndX * SX - PX + (KX - 1);
            const int inBegX = MAX(inBegX0, 0);
            const int inEndX = MIN(inEndX0, IW - 1);
            const int inSizeX = (tilingMode == nn::shave_lib::TilingCW) ? (inEndX - inBegX + 1) : IW;

            const int begPX = inBegX - inBegX0;
            const int endPX = inEndX0 - inEndX;

            { // fill pad areas with 0s
                const int alignSizeL = ALIGN_TO_MULTIPLE(VECTOR_SIZE, begPX);
                const int alignSizeR = ALIGN_TO_MULTIPLE(VECTOR_SIZE, endPX);

                const int inLines = inSizeY * numChannels;
                const int outLines = outSizeY * numChannels;

                if (rowsFirst) {
                    const int inSizeXA = ALIGN_TO_MULTIPLE(VECTOR_SIZE, inSizeX);
                    const int alignSizeV = ALIGN_TO_MULTIPLE(VECTOR_SIZE, inSizeXA - inSizeX);
                    const int offsetV = inSizeXA - alignSizeV;

                    const int rowVectors = DIVR(inSizeX, VECTOR_SIZE);
                    const int inStepY = rowVectors * VECTOR_SIZE;
                    const int tmpStepY = begPX + MAX(inStepY, inSizeX + endPX);

                    const int offsetL = 0;
                    const int offsetR = MAX(0, tmpStepY - alignSizeR);

                    if (alignSizeV > 0)
                        CHW_fill(inoutBuffer + offsetV, inLines, inSizeXA, alignSizeV);
                    if (begPX > 0)
                        CHW_fill(tmpBuffer + offsetL, outLines, tmpStepY, alignSizeL);
                    if (endPX > 0)
                        CHW_fill(tmpBuffer + offsetR, outLines, tmpStepY, alignSizeR);
                } else {
                    const int inSizeXPP = rowsFirst ? (begPX + inSizeX + endPX) : ((outSizeX - 1) * SX + KX);

                    const int offsetL = 0;
                    const int offsetR = MAX(0, inSizeXPP - alignSizeR);

                    if (begPX > 0)
                        CHW_fill(inoutBuffer + offsetL, inLines, inSizeXPP, alignSizeL);
                    if (endPX > 0)
                        CHW_fill(inoutBuffer + offsetR, inLines, inSizeXPP, alignSizeR);
                }
            }

            fp16 *src = ((fp16 *)input) + ((begChannel * IH + inBegY) * IS + inBegX);
            fp16 *dst = rowsFirst ? inoutBuffer : (inoutBuffer + begPX);
            int byteLength = inSizeY * inSizeX * numChannels * INPUT_BPP;
            int srcWidth = 0, srcStride = 0;
            int dstWidth = inSizeX * INPUT_BPP;
            int dstStride = (rowsFirst ? ALIGN_TO_MULTIPLE(VECTOR_SIZE, inSizeX) : ((outSizeX - 1) * SX + KX)) * INPUT_BPP;

            switch (tilingMode) {
            case nn::shave_lib::TilingC:
                srcWidth = inSizeX * INPUT_BPP;
                srcStride = IS * INPUT_BPP;
                break;
            case nn::shave_lib::TilingCH:
                srcWidth = inSizeY * IW * INPUT_BPP;
                srcStride = IH * IW * INPUT_BPP;
                break;
            case nn::shave_lib::TilingCW:
                srcWidth = inSizeX * INPUT_BPP;
                srcStride = IS * INPUT_BPP;
                break;
            }
            dmaStart(dma1, src, dst, byteLength, srcWidth, dstWidth, srcStride, dstStride);
            dma1.wait();

            if (rowsFirst) {
                const int rowVectors = DIVR(inSizeX, VECTOR_SIZE);
                const int inStepY = rowVectors * VECTOR_SIZE;
                const int tmpStepY = begPX + MAX(inStepY, inSizeX + endPX);

                sumRow(inoutBuffer, tmpBuffer + begPX, numChannels, rowVectors, inBegY, inSizeY, outBegY, outSizeY, inStepY,
                       tmpStepY, p);
                sumCol(tmpBuffer + begPX, inoutBuffer, numChannels, outSizeY, inBegX, outBegX, outSizeX, tmpStepY, outSizeX,
                       p);
            } else {
                const int rowVectors = DIVR(outSizeX, VECTOR_SIZE);
                const int tmpStepY = rowVectors * VECTOR_SIZE;
                const int inStepY = (outSizeX - 1) * SX + KX;

                sumCol(inoutBuffer + begPX, tmpBuffer, numChannels, inSizeY, inBegX, outBegX, outSizeX, inStepY, tmpStepY,
                       p);
                sumRow(tmpBuffer, inoutBuffer, numChannels, rowVectors, inBegY, inSizeY, outBegY, outSizeY, tmpStepY,
                       outSizeX, p);
            }

            if (fillInAvg) {
                calcScales(inBegY, inSizeY, outBegY, outSizeY, KY, SY, PY, rowScales);
                calcScales(inBegX, inSizeX, outBegX, outSizeX, KX, SX, PX, colScales);
                CHW_scaleAvg(inoutBuffer, outSizeY, outSizeX, numChannels, rowScales, colScales);
            } else {
                const half8 areaScale = (half8)(1.0f / float(KY * KX));
                CHW_scaleZero(inoutBuffer, outSizeY, outSizeX, numChannels, areaScale);
            }

            src = inoutBuffer;
            dst = output + ((begChannel * OH + outBegY) * OS + outBegX);
            byteLength = outSizeY * outSizeX * numChannels * INPUT_BPP;
            srcWidth = outSizeY * outSizeX * numChannels * INPUT_BPP;
            srcStride = outSizeY * outSizeX * numChannels * INPUT_BPP;
            switch (tilingMode) {
            case nn::shave_lib::TilingC:
                dstWidth = outSizeX * INPUT_BPP;
                dstStride = OS * INPUT_BPP;
                break;
            case nn::shave_lib::TilingCH:
                dstWidth = outSizeY * OW * INPUT_BPP;
                dstStride = OH * OW * INPUT_BPP;
                break;
            case nn::shave_lib::TilingCW:
                dstWidth = outSizeX * INPUT_BPP;
                dstStride = OS * INPUT_BPP;
                break;
            }
            dmaStart(dma1, src, dst, byteLength, srcWidth, dstWidth, srcStride, dstStride);
            dma1.wait();
        }
    }
}

void HWC_mvAvgPool3x3_runner(nn::shave_lib::PoolParams *p) {
    const half * in = p->input;
    half * out = p->output;
    for (uint32_t n = 0; n < p->batch_dim; ++n)
    {
        u32 inputStart  = (uint32_t)(in  + (p->input_batch_step * n));
        u32 outputStart = (uint32_t)(out + (p->output_batch_step * n));

        u32 channelsPerGroup = p->channelsPerGroup;
        u32 maxGroupsPerShave = p->maxGroupsPerShave;
        u32 totalGroupsRemain = p->sliceGroups;
        u32 offsetIO = 0;
        DmaAlShave dmaTask_;

        do {
            u32 groups = MIN(maxGroupsPerShave, totalGroupsRemain);
            p->sliceChannels = groups * channelsPerGroup;
            p->input = (fp16 *)(inputStart + offsetIO * sizeof(fp16));
            p->output = (fp16 *)(outputStart + offsetIO * sizeof(fp16));

            HWC_mvAvgPool3x3(p, &dmaTask_);

            offsetIO += p->sliceChannels;
            totalGroupsRemain -= groups;
        } while (totalGroupsRemain);
    }
}

void HWC_mvAvgPool3x3(nn::shave_lib::PoolParams *p, DmaAlShave *dmaTask_) {
    u32 H = p->inHeight, W = p->inWidth, WP, HP, C = p->channels;
    u32 Hout = 0, Wout = 0;
    u32 sliceC = p->sliceChannels;
    u32 strideX = p->strideX, strideY = p->strideX;
    u32 ostrideX = p->outputStride;
    u32 i, j;
    u8 *inAddress = (u8 *)((u32)p->input);
    u8 *linesBuffer;
    u8 *outputBuffer;
    half *inLines[3];
    half *kernelInLines[3];
    half *kernelOutLines[1];
    u32 writtenElems = 0, readElems = 0;
    u32 line = 0;

    // user specified padding (used only for CAFFE-style padding)
    u32 padUserX = p->padX, padUserY = p->padX;
    // convenience padding; used to simplify computations
    s32 padConvLeft, padConvRight, padConvTop, padConvBottom;
    s32 padLeft, padRight, padTop, padBottom, Hpad, Wpad;
    u32 numElem;

    // set buffers to point to locations relative to cmxslice
    // the memory allocation is done in MatMul (84000 bytes/slice)
    linesBuffer = p->cmxslice;
    outputBuffer = p->cmxslice + p->availableCmxBytes / 2;

    {
        // rules for CAFFE padding
        Hout = ((H + 2 * padUserY - 3 + strideY - 1) / strideY) + 1;
        Wout = ((W + 2 * padUserX - 3 + strideX - 1) / strideX) + 1;
        Hout = MIN(Hout, (H + padUserY + strideY - 1) / strideY);
        Wout = MIN(Wout, (W + padUserX + strideX - 1) / strideX);

        Hpad = ((Hout - 1) * strideY + 3 - H - 2 * padUserY);
        Wpad = ((Wout - 1) * strideX + 3 - W - 2 * padUserX);
        Hpad = MAX(0, Hpad);
        Wpad = MAX(0, Wpad);

        padConvLeft = 0;
        padConvRight = Wpad - padConvLeft;
        padConvTop = 0;
        padConvBottom = Hpad - padConvTop;
    }

    padLeft = padUserX + padConvLeft;
    padRight = padUserX + padConvRight;
    padTop = padUserY + padConvTop;
    padBottom = padUserY + padConvBottom;

    HP = H + padTop + padBottom;
    WP = W + padLeft + padRight;

    for (i = padTop; i < 3; i++) {
        dmaTask_->start(inAddress + readElems * C * INPUT_BPP,                 // src
                        linesBuffer + (i * WP + padLeft) * sliceC * INPUT_BPP, // dst
                        W * sliceC * INPUT_BPP,                                // byte length
                        sliceC * INPUT_BPP,                                    // src width
                        sliceC * INPUT_BPP,                                    // dst width
                        C * INPUT_BPP,                                         // src stride
                        sliceC * INPUT_BPP);                                   // dst stride
        readElems += W;
    }

    inLines[0] = (half *)linesBuffer + padLeft * sliceC;
    inLines[1] = (half *)linesBuffer + (WP + padLeft) * sliceC;
    inLines[2] = (half *)linesBuffer + (2 * WP + padLeft) * sliceC;

    // Padding: padding one row in column-major is equivalent with padding one column in row-major
    //          top row -> leftmost column, bottom row -> rightmost column

    // left padding;
    for (i = 0; i < sliceC * padUserX; i++) {
        *(inLines[0] - padUserX * sliceC + i) = 0;
        *(inLines[1] - padUserX * sliceC + i) = 0;
        *(inLines[2] - padUserX * sliceC + i) = 0;
    }
    // padConvLeft can only be 0 or 1
    for (i = 0; i < padConvLeft * sliceC; i++) {
        *(inLines[0] - padLeft * sliceC + i) =
            (*(inLines[0] + (-padLeft + 1) * sliceC + i) + *(inLines[0] + (-padLeft + 2) * sliceC + i)) / 2;
        *(inLines[1] - padLeft * sliceC + i) =
            (*(inLines[1] + (-padLeft + 1) * sliceC + i) + *(inLines[1] + (-padLeft + 2) * sliceC + i)) / 2;
        *(inLines[2] - padLeft * sliceC + i) =
            (*(inLines[2] + (-padLeft + 1) * sliceC + i) + *(inLines[2] + (-padLeft + 2) * sliceC + i)) / 2;
    }

    // right padding
    for (i = 0; i < sliceC * padUserX; i++) {
        *(inLines[0] + W * sliceC + i) = 0;
        *(inLines[1] + W * sliceC + i) = 0;
        *(inLines[2] + W * sliceC + i) = 0;
    }
    // padConvRight can only be 0 or 1
    for (i = 0; i < padConvRight * sliceC; i++) {
        *(inLines[0] + (W + padRight - 1) * sliceC + i) =
            (*(inLines[0] + (W + padRight - 2) * sliceC + i) + *(inLines[0] + (W + padRight - 3) * sliceC + i)) / 2;
        *(inLines[1] + (W + padRight - 1) * sliceC + i) =
            (*(inLines[1] + (W + padRight - 2) * sliceC + i) + *(inLines[1] + (W + padRight - 3) * sliceC + i)) / 2;
        *(inLines[2] + (W + padRight - 1) * sliceC + i) =
            (*(inLines[2] + (W + padRight - 2) * sliceC + i) + *(inLines[2] + (W + padRight - 3) * sliceC + i)) / 2;
    }

    // top padding; user pad is zero for any padding scheme
    for (j = (u32)padConvTop; j < (u32)padTop; j++) {
        for (i = 0; i < sliceC * WP; i++)
            *(inLines[j] + -padLeft * sliceC + i) = 0;
    }
    // padConvTop is either 0 or 1
    if (padConvTop == 1) {
        for (i = 0; i < sliceC * WP; i++) {
            *(inLines[0] - padLeft * sliceC + i) =
                (*(inLines[1] - padLeft * sliceC + i) + *(inLines[2] - padLeft * sliceC + i)) / 2;
        }
    }

    kernelInLines[0] = (half *)linesBuffer;
    kernelInLines[1] = (half *)linesBuffer + 1 * sliceC;
    kernelInLines[2] = (half *)linesBuffer + 2 * sliceC;

    kernelOutLines[0] = (half *)outputBuffer;

    // align numElem to the next multiple of 8, as this is a constraint of the
    // asm implementation of mvcvMaximumV3_asm
    // this works because sliceC was chosen such that there are at least
    // 7*INPUT_BPP bytes at the end of the input/output buffers
    numElem = (sliceC * 3 * WP + 7) & ~0x7;
    mvcvAverageV3_asm((half **)kernelInLines, (half **)kernelOutLines, numElem);

    kernelInLines[0] = (half *)outputBuffer;
    kernelInLines[1] = (half *)outputBuffer + WP * sliceC;
    kernelInLines[2] = (half *)outputBuffer + 2 * WP * sliceC;

    // again, align numElem to the next multiple of 8
    numElem = (sliceC * W + 7) & ~0x7;
    mvcvAverageV3_asm((half **)kernelInLines, (half **)kernelOutLines, numElem);

    dmaTask_->start(outputBuffer,                 // src
                    (u8 *)p->output,              // dst
                    Wout * sliceC * INPUT_BPP,    // byte length
                    sliceC * INPUT_BPP,           // src width
                    sliceC * INPUT_BPP,           // dst width
                    strideX * sliceC * INPUT_BPP, // src stride
                    ostrideX);                    // dst stride
    dmaTask_->wait();

    writtenElems += Wout;

    line = 1;
    do {
        u32 bufferIdx = (line - 1);
        if (readElems < W * H) {
            dmaTask_->start(inAddress + readElems * C * INPUT_BPP,                               // src
                            linesBuffer + ((bufferIdx % 3) * WP + padLeft) * sliceC * INPUT_BPP, // dst
                            W * sliceC * INPUT_BPP,                                              // byte length
                            sliceC * INPUT_BPP,                                                  // src width
                            sliceC * INPUT_BPP,                                                  // dst width
                            C * INPUT_BPP,                                                       // src stride
                            sliceC * INPUT_BPP);                                                 // dst stride
            dmaTask_->wait();

            readElems += W;
            inLines[2] = (half *)(linesBuffer + ((bufferIdx % 3) * WP + padLeft) * sliceC * INPUT_BPP);

            // left padding;
            for (i = 0; i < sliceC * padUserX; i++) {
                *(inLines[2] - padUserX * sliceC + i) = 0;
            }
            // padConvLeft can only be 0 or 1
            for (i = 0; i < padConvLeft * sliceC; i++) {
                *(inLines[2] - padLeft * sliceC + i) =
                    (*(inLines[2] + (-padLeft + 1) * sliceC + i) + *(inLines[2] + (-padLeft + 2) * sliceC + i)) / 2;
            }

            // right padding
            for (i = 0; i < sliceC * padUserX; i++) {
                *(inLines[2] + W * sliceC + i) = 0;
            }
            // padConvRight can only be 0 or 1
            for (i = 0; i < padConvRight * sliceC; i++) {
                *(inLines[2] + (W + padRight - 1) * sliceC + i) = (*(inLines[2] + (W + padRight - 2) * sliceC + i) +
                                                                   *(inLines[2] + (W + padRight - 3) * sliceC + i)) /
                                                                  2;
            }
        } else if (padBottom) {
            s32 padLine = line + 3 - 1 - padTop - H;
            // bottom padding; user pad is zero for any padding scheme
            if (padLine < (s32)padUserY) {
                for (i = 0; i < sliceC * WP; i++) {
                    *((half *)linesBuffer + (bufferIdx % 3) * WP * sliceC + i) = 0;
                }
            } else if (padLine == (s32)padUserY) {
                // padConvBottom is either 0 or 1
                for (i = 0; i < sliceC * WP; i++) {
                    *((half *)linesBuffer + ((bufferIdx % 3) * WP) * sliceC + i) =
                        (*((half *)linesBuffer + (((bufferIdx + 1) % 3) * WP) * sliceC + i) +
                         *((half *)linesBuffer + (((bufferIdx + 2) % 3) * WP) * sliceC + i)) /
                        2;
                }
            }
        }

        if (line % strideY == 0) {
            kernelInLines[0] = (half *)linesBuffer;
            kernelInLines[1] = (half *)linesBuffer + 1 * sliceC;
            kernelInLines[2] = (half *)linesBuffer + 2 * sliceC;
            // again, align numElem to the next multiple of 8
            numElem = (sliceC * 3 * WP + 7) & ~0x7;
            mvcvAverageV3_asm((half **)kernelInLines, (half **)kernelOutLines, numElem);

            kernelInLines[0] = (half *)outputBuffer;
            kernelInLines[1] = (half *)outputBuffer + WP * sliceC;
            kernelInLines[2] = (half *)outputBuffer + 2 * WP * sliceC;

            // again, align numElem to the next multiple of 8
            numElem = (sliceC * W + 7) & ~0x7;
            mvcvAverageV3_asm((half **)kernelInLines, (half **)kernelOutLines, numElem);

            dmaTask_->start(outputBuffer,                                              // src
                            (u8 *)(p->output + writtenElems * (ostrideX / INPUT_BPP)), // dst
                            Wout * sliceC * INPUT_BPP,                                 // byte length
                            sliceC * INPUT_BPP,                                        // src width
                            sliceC * INPUT_BPP,                                        // dst width
                            strideX * sliceC * INPUT_BPP,                              // src stride
                            ostrideX);                                                 // dst stride
            writtenElems += Wout;
        }

        line++;
    } while (writtenElems < (Wout * Hout));

    dmaTask_->wait();
}

#endif

void HWC_mvAvgPoolMxN(nn::act_shave_lib::PoolParams *p);
void CHW_mvAvgPoolMxN(nn::act_shave_lib::PoolParams *p);
void HWC_mvAvgPool3x3_runner(nn::act_shave_lib::PoolParams *p);

extern "C" {
void AvgPoolingKernel(nn::act_shave_lib::PoolParams *pp) {
    if (pp->tilingMode == TilingUnknown || pp->numTiles == 0) return;

    switch (pp->mode) {
    case HWC_AvgPoolMxN: HWC_mvAvgPoolMxN(pp); break;
    case CHW_AvgPoolMxN: CHW_mvAvgPoolMxN(pp); break;
    case HWC_AvgPool3x3: HWC_mvAvgPool3x3_runner(pp); break;
    }
}
}
