///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
/// @file      sippDefs.h
/// 

#ifndef _SIPP_DEFS_H_
#define _SIPP_DEFS_H_

#include "sippBaseTypes.h"
#include "sippHwIds.h"

#define N_PL(x)   (x)       // Number of planes.
#define SZ(x)     sizeof(x) // Size of x.
#define SIPP_AUTO (-1)

// SIPP Filter Specific macros
#define SIPP_MAX_WARP_CTXS (3)

////////////////////////////////////////////////////////////////
// Sipp Filter Flags (SippFilter.flags)
#define SIPP_REQ_SW_VIEW         (1<<2)
#define SIPP_FLAG_DO_H_PADDING   (1<<3)
#define SIPP_RESIZE              (1<<4)
#define SIPP_CROP                (1<<5)
#define SIPP_SHIFT_PLANES        (1<<6)
#define SIPP_PROVIDE_CHUNK_POS   (1<<7)
#define SIPP_RESIZE_OPEN_CV      ((1<<8) | SIPP_RESIZE)
#define SIPP_ANCHOR              (1<<9)
#define SIPP_UNALIGN_SLICE_WIDTH (1<<10)
#define SIPP_EVEN_SLICE_WIDTH    (1<<11)

#define SIPP_MBIN(x) x

typedef int32_t FnSvuRun;

#define SVU_SYM(s) s

#define BASE_SW_ID (SIPP_MAX_HW_ID + 1)

enum SippFilters {
    svuAbsdiff                            = BASE_SW_ID,
    svuAddV2Fp16                          = BASE_SW_ID + 1,
    svuArithmeticAdd                      = BASE_SW_ID + 2,
    svuArithmeticAddmask                  = BASE_SW_ID + 3,
    svuArithmeticSub                      = BASE_SW_ID + 4,
    svuArithmeticSubFp16ToFp16            = BASE_SW_ID + 5,
    svuArithmeticSubmask                  = BASE_SW_ID + 6,
    svuArithmeticSubU16                   = BASE_SW_ID + 7,
    svuBitwiseAnd                         = BASE_SW_ID + 8,
    svubitwiseAndMask                     = BASE_SW_ID + 9,
    svuBitwiseNot                         = BASE_SW_ID + 10,
    svuBitwiseOr                          = BASE_SW_ID + 11,
    svuBitwiseOrMask                      = BASE_SW_ID + 12,
    svuBitwiseXor                         = BASE_SW_ID + 13,
    svuBitwiseXorMask                     = BASE_SW_ID + 14,
    svuBoxFilter                          = BASE_SW_ID + 15,
    svuBoxFilter11x11                     = BASE_SW_ID + 16,
    svuBoxFilter13x13                     = BASE_SW_ID + 17,
    svuBoxFilter15x15                     = BASE_SW_ID + 18,
    svuBoxFilter3x3                       = BASE_SW_ID + 19,
    svuBoxFilter5x5                       = BASE_SW_ID + 20,
    svuBoxFilter7x7                       = BASE_SW_ID + 21,
    svuBoxFilter9x9                       = BASE_SW_ID + 22,
    svuCensusTransform5x5                 = BASE_SW_ID + 23,
    svuCensusTransformAverageRefMask7x7   = BASE_SW_ID + 24,
    svuChannelExtract                     = BASE_SW_ID + 25,
    svuConvertF16ToU8                     = BASE_SW_ID + 26,
    svuConvertU8ToF16                     = BASE_SW_ID + 27,
    svuConvertYUV400ToYUV422              = BASE_SW_ID + 28,
    svuConv1x5                            = BASE_SW_ID + 29,
    svuConv1x5Fp16ToFp16                  = BASE_SW_ID + 30,
    svuConv1x7Fp16ToFp16                  = BASE_SW_ID + 31,
    svuConv3x3                            = BASE_SW_ID + 32,
    svuConvolution3x3s2hhhh               = BASE_SW_ID + 33,
    svuConvolution3x3s3hhhh               = BASE_SW_ID + 34,
    svuConvolution3x3s4hhhh               = BASE_SW_ID + 35,
    svuConvolution3x3s8hhhh               = BASE_SW_ID + 36,
    svuConv5x1                            = BASE_SW_ID + 37,
    svuConv5x1Fp16ToFp16                  = BASE_SW_ID + 38,
    svuConv5x5                            = BASE_SW_ID + 39,
    svuConvolution5x5s2hhhh               = BASE_SW_ID + 40,
    svuConvolution5x5s3hhhh               = BASE_SW_ID + 41,
    svuConvolution5x5s4hhhh               = BASE_SW_ID + 42,
    svuConvolution5x5s8hhhh               = BASE_SW_ID + 43,
    svuConv7x1                            = BASE_SW_ID + 44,
    svuConv7x7Fp16ToU8                    = BASE_SW_ID + 45,
    svuConvolution7x7s2hhhh               = BASE_SW_ID + 46,
    svuConvolution7x7s4hhhh               = BASE_SW_ID + 47,
    svuConvolution7x7s8hhhh               = BASE_SW_ID + 48,
    svuConvSeparable3x3                   = BASE_SW_ID + 49,
    svuConvSeparable3x3Fp16ToFp16         = BASE_SW_ID + 50,
    svuConvSeparable5x5                   = BASE_SW_ID + 51,
    svuConvSeparable5x5Fp16ToFp16         = BASE_SW_ID + 52,
    svuConvSeparable7x7                   = BASE_SW_ID + 53,
    svuConvSeparable7x7Fp16ToFp16         = BASE_SW_ID + 54,
    svuConvSeparable9x9                   = BASE_SW_ID + 55,
    svucvtColorNV12toBGR                  = BASE_SW_ID + 56,
    svucvtColorNV12toRGB                  = BASE_SW_ID + 57,
    svucvtColorNV21toRGB                  = BASE_SW_ID + 58,
    svuCvtColorRGBtoLuma                  = BASE_SW_ID + 59,
    svuCvtColorRGBToYUV422                = BASE_SW_ID + 60,
    svuDilate7x7                          = BASE_SW_ID + 61,
    svuErode3x3                           = BASE_SW_ID + 62,
    svuErode5x5                           = BASE_SW_ID + 63,
    svuGauss                              = BASE_SW_ID + 64,
    svuGaussHx2                           = BASE_SW_ID + 65,
    svuGaussHx2_fp16                      = BASE_SW_ID + 66,
    svuGaussVx2                           = BASE_SW_ID + 67,
    svuGaussVx2_fp16                      = BASE_SW_ID + 68,
    svuHistogramStat                      = BASE_SW_ID + 69,
    svuIntegralImageSqSumU32M2            = BASE_SW_ID + 70,
    svuIntegralImageSumU16U32             = BASE_SW_ID + 71,
    svuLaplacian3x3                       = BASE_SW_ID + 72,
    svuLaplacian5x5                       = BASE_SW_ID + 73,
    svuLaplacian7x7                       = BASE_SW_ID + 74,
    svuMedianFilter11x11                  = BASE_SW_ID + 75,
    svuMedianFilter13x13                  = BASE_SW_ID + 76,
    svuMedianFilter15x15                  = BASE_SW_ID + 77,
    svuMedianFilter3x3                    = BASE_SW_ID + 78,
    svuMedianFilter5x5                    = BASE_SW_ID + 79,
    svuMedianFilter7x7                    = BASE_SW_ID + 80,
    svuMedianFilter9x9                    = BASE_SW_ID + 81,
    svuPositionKernel                     = BASE_SW_ID + 82,
    svuPyrDown                            = BASE_SW_ID + 83,
    svuSAD11x11                           = BASE_SW_ID + 84,
    svuSAD5x5                             = BASE_SW_ID + 85,
    svuScale05BilinHV_Fp16U8              = BASE_SW_ID + 86,
    svuScale05BilinHVFp16                 = BASE_SW_ID + 87,
    svuScale2xBilinHV_Fp16U8_phase025_075 = BASE_SW_ID + 88,
    svuScale2xBilinHV_U8ToU8_phase025_075 = BASE_SW_ID + 89,
    svuScaleBilinearPlanar                = BASE_SW_ID + 90,
    svuScl05Lanc6                         = BASE_SW_ID + 91,
    svuScharr_fp16                        = BASE_SW_ID + 92,
    svuSobel                              = BASE_SW_ID + 93,
    svuSSD11x11                           = BASE_SW_ID + 94,
    svuSSD5x5                             = BASE_SW_ID + 95,
    svuSSD7x7U8ToU32                      = BASE_SW_ID + 96,
    svuSsdPointLine7x7U8U32               = BASE_SW_ID + 97,
    svuThreshold                          = BASE_SW_ID + 98,
    svuThresholdBinaryRange               = BASE_SW_ID + 99,
    svuThresholdBinaryU8                  = BASE_SW_ID + 100,
    svucvtColorNV12toRGBi                 = BASE_SW_ID + 101,
    svucvtColorNV12toBGRi                 = BASE_SW_ID + 102,
    svucvtColorNV21toRGBi                 = BASE_SW_ID + 103,
    svucvtColorNV21toBGRi                 = BASE_SW_ID + 104,
    svuScaleBilinear                      = BASE_SW_ID + 105,
    svuMaximumV9x4                        = BASE_SW_ID + 106,
    svuCvtColorRGBtoUV                    = BASE_SW_ID + 107,
    svuDownSampleBilinearLine             = BASE_SW_ID + 108,
    svuScale2xBilinHV_odd_even_Fp16ToFp16 = BASE_SW_ID + 109,
    svuResizeAlphaFp16ToFp16              = BASE_SW_ID + 110,
    svuCvtColorRGBtoLumaNV12              = BASE_SW_ID + 111,
    svuCvtColorRGBtoChromaNV12            = BASE_SW_ID + 112,
    svuCvtColorRGBtoUV420                 = BASE_SW_ID + 113,
    svuCensusTransform11x11               = BASE_SW_ID + 114,
    svuCensusTransform11x11u8             = BASE_SW_ID + 115,
    svuCensusTransform7x7                 = BASE_SW_ID + 116,
    svuCensusTransformAverageRef7x7       = BASE_SW_ID + 117,
    svuConvolution3x3s2xhhx               = BASE_SW_ID + 118,
    svuGauss1x5_u16in_u32out              = BASE_SW_ID + 119,
    svuGauss5x1_u32in_u16out              = BASE_SW_ID + 120,
    svuIntegralImageSumF32M2              = BASE_SW_ID + 121,
    svuIntegralImageSumU32M2              = BASE_SW_ID + 122,
    svuMaximumV2                          = BASE_SW_ID + 123,
    svuMaximumV3                          = BASE_SW_ID + 124,
    svuMaximumV9                          = BASE_SW_ID + 125,
    svuScaleFp16                          = BASE_SW_ID + 126,
    svuConvolution7x7s3hhhh               = BASE_SW_ID + 127,
    svuConv3x3Fp16ToFp16                  = BASE_SW_ID + 128,
    svuConv5x5Fp16ToFp16                  = BASE_SW_ID + 129,
    svuConv7x1Fp16ToFp16                  = BASE_SW_ID + 130,
    svuConv7x7Fp16ToFp16                  = BASE_SW_ID + 131,
    svuCvtColorYUV422ToRGB                = BASE_SW_ID + 132,
    svuConv1x7                            = BASE_SW_ID + 133,
    svuConv1x9                            = BASE_SW_ID + 134,
    svuConv7x7                            = BASE_SW_ID + 135,
    svuConv1x15                           = BASE_SW_ID + 136,
    svuConv9x1                            = BASE_SW_ID + 137,
    svuConv15x1                           = BASE_SW_ID + 138,
    svuIntegralImageSqSumF32M2            = BASE_SW_ID + 139,
    svuLaplacian5x5Fp16ToFp16             = BASE_SW_ID + 140,
    svuLaplacian7x7Fp16ToFp16             = BASE_SW_ID + 141,
    svuMatAdd                             = BASE_SW_ID + 142,
    svuMatMinus                           = BASE_SW_ID + 143,
    svuScl05BilinHV                       = BASE_SW_ID + 144,
    svuDilate5x5                          = BASE_SW_ID + 145,
    svuGenNoiseFp16                       = BASE_SW_ID + 146,
    svuScale2xBilinHV_025_075_U16ToU16    = BASE_SW_ID + 147,
    svusLaplacian3x3Fp16ToFp16            = BASE_SW_ID + 148,
    svuMerge3p                            = BASE_SW_ID + 149,
    svuScale2xBilinHV_025_075_Fp16ToFp16  = BASE_SW_ID + 150,
    svuErode7x7                           = BASE_SW_ID + 151,
    svuBilinearInterpolation              = BASE_SW_ID + 152,
    svuMinMaxPos                          = BASE_SW_ID + 153,
    svuMeanStdDev                         = BASE_SW_ID + 154,
    svuMinMaxValue                        = BASE_SW_ID + 155,
    svuPadKernel_u8                       = BASE_SW_ID + 156,
    svuPadKernel_u16                      = BASE_SW_ID + 157,
    svuCvtColorYUVToRGB                   = BASE_SW_ID + 158,
    svuDilate3x3                          = BASE_SW_ID + 159,
    svuCvtColorNV12toYUV422i              = BASE_SW_ID + 160,
    svuCalcEpipolarDistance               = BASE_SW_ID + 161,
    svuCalcG                              = BASE_SW_ID + 162,
    svuCalcBxBy                           = BASE_SW_ID + 163,
    svuCvtColorChromaYUVToNV12            = BASE_SW_ID + 164,
    svuCvtColorRGBfp16ToLumaU8            = BASE_SW_ID + 165,
    svuCvtColorRGBfp16ToUV420U8           = BASE_SW_ID + 166,
    svuHammingDistance                    = BASE_SW_ID + 167,
    svuMerge2p                            = BASE_SW_ID + 168,
    svuMerge4p                            = BASE_SW_ID + 169,
    svuSplit2p                            = BASE_SW_ID + 170,
    svuSplit3p                            = BASE_SW_ID + 171,
    svuSplit4p                            = BASE_SW_ID + 172,
    svuConvSeparable11x11                 = BASE_SW_ID + 173,
    svuConvSeparable11x11Fp16ToFp16       = BASE_SW_ID + 174,
    svuConvSeparable9x9Fp16ToFp16         = BASE_SW_ID + 175,
    svuConvolution11x11s3hhhh             = BASE_SW_ID + 176,
    svuConvolution11x11s4hhhh             = BASE_SW_ID + 177,
    svuConvolution11x11s8hhhh             = BASE_SW_ID + 178,
    svuConv11x11                          = BASE_SW_ID + 179,
    svuConvolution11x11s1hhhh             = BASE_SW_ID + 180,
    svuConvolution11x11s1xxhx             = BASE_SW_ID + 181,
    svuConvolution11x11s2hhhh             = BASE_SW_ID + 182,
    svuConvolution11x11s2xxhx             = BASE_SW_ID + 183,
    svuConv3x3fp32Scharr                  = BASE_SW_ID + 184,
    svuCvtColorRGBtoNV21                  = BASE_SW_ID + 185,
    svuConv9x9                            = BASE_SW_ID + 186,
    svuConv9x9Fp16ToFp16                  = BASE_SW_ID + 187,
    svuConvolution9x9s2hhhh               = BASE_SW_ID + 188,
    svuConvolution9x9s3hhhh               = BASE_SW_ID + 189,
    svuConvolution9x9s4hhhh               = BASE_SW_ID + 190,
    svuConvolution9x9s8hhhh               = BASE_SW_ID + 191,
    svuCvtColorKernelRGBtoYUV             = BASE_SW_ID + 192,
    svuBilateral5x5                       = BASE_SW_ID + 193,
    svuHistogram                          = BASE_SW_ID + 194,
    svuEqualizeHist                       = BASE_SW_ID + 195,
    svuDilateGeneric                      = BASE_SW_ID + 196,
    svuCvtInterleavedtoPlanar             = BASE_SW_ID + 197,
    svuGauss3x3                           = BASE_SW_ID + 198,
    svuDrop4Channel                       = BASE_SW_ID + 199,
};

// SIPP function identifiers.
enum class SippMessageType {
    CREATE_PLUGIN = 0,  // Create FLIC SIPP plugin
    SIPP_CUSTOM_KERNEL, // SIPP custom kernel.
    INITMASK,           // Set initialisation mask.
    BPCREATE,           // Basic pipeline create.
    PCREATE,            // Pipeline create.
    ADDRESOURCE,        // Add pipeline resource.
    PFINALIZE,          // Finalize pipeline.
    FCREATE,            // Filter create.
    FSETFSZ,            // Filter set frame size
    FCONFIG,            // Filter configuration.
    SETBITSPP,          // Set bits per pixel.
    FLINK,              // Filter link.
    FLINKSETOBUF,       // Filter link set output buffer.
    FADDOBUF,           // Filter add output buffer.
    PDELETE,            // Delete pipeline.
    REGCB,              // Register callback.
    PROCFRAMENB,        // Process frame (non-blocking).
    LINESITER,          // Number of lines per iteration.
    LASTERROR,          // Last error.
    ERRORHIST,          // Error history.
    ERRORSTAT,          // Error status.
    GETPORT,            // Get port number fo a specific dma filter
};

//////////////////////////////////////////////////////////////
// SIPP STATUS enumeration
typedef enum
{
    /*   0  0x00  */   eSIPP_STATUS_OK = 0,
    /*   1  0x01  */   eSIPP_STATUS_ALREADY_INIT,
    /*   2  0x02  */   eSIPP_STATUS_NOT_INIT,
    /*   3  0x03  */   eSIPP_STATUS_INTERNAL_ERROR,
    /*   4  0x04  */   eSIPP_STATUS_BAD_HANDLE,
    /*   5  0x05  */   eSIPP_STATUS_BAD_PARAMETER,
    /*   6  0x06  */   eSIPP_STATUS_BAD_LENGTH,
    /*   7  0x07  */   eSIPP_STATUS_BAD_UNIT,
    /*   8  0x08  */   eSIPP_STATUS_RESOURCE_ERROR,
    /*   9  0x09  */   eSIPP_STATUS_CLOSED_HANDLE,
    /*  10  0x0A  */   eSIPP_STATUS_TIMEOUT,
    /*  11  0x0B  */   eSIPP_STATUS_NOT_ATTACHED,
    /*  12  0x0C  */   eSIPP_STATUS_NOT_SUPPORTED,
    /*  13  0x0D  */   eSIPP_STATUS_REOPENED_HANDLE,
    /*  14  0x0E  */   eSIPP_STATUS_INVALID,
    /*  15  0x0F  */   eSIPP_STATUS_DESTROYED,
    /*  16  0x10  */   eSIPP_STATUS_DISCONNECTED,
    /*  17  0x11  */   eSIPP_STATUS_BUSY,
    /*  18  0x12  */   eSIPP_STATUS_IN_USE,
    /*  19  0x13  */   eSIPP_STATUS_CANCELLED,
    /*  20  0x14  */   eSIPP_STATUS_UNDEFINED,
    /*  21  0x15  */   eSIPP_STATUS_UNKNOWN,
    /*  22  0x16  */   eSIPP_STATUS_NOT_FOUND,
    /*  23  0x17  */   eSIPP_STATUS_NOT_AVAILABLE,
    /*  24  0x18  */   eSIPP_STATUS_NOT_COMPATIBLE,
    /*  25  0x19  */   eSIPP_STATUS_NOT_IMPLEMENTED,
    /*  26  0x1A  */   eSIPP_STATUS_EMPTY,
    /*  27  0x1B  */   eSIPP_STATUS_FULL,
    /*  28  0x1C  */   eSIPP_STATUS_FAILURE,
    /*  29  0x1D  */   eSIPP_STATUS_ALREADY_ATTACHED,
    /*  30  0x1E  */   eSIPP_STATUS_ALREADY_DONE,
    /*  31  0x1F  */   eSIPP_STATUS_ASLEEP,
    /*  32  0x20  */   eSIPP_STATUS_BAD_ATTACHMENT,
    /*  33  0x21  */   eSIPP_STATUS_BAD_COMMAND,
    /*  34  0x22  */   eSIPP_STATUS_INT_HANDLED,
    /*  35  0x23  */   eSIPP_STATUS_INT_NOT_HANDLED,
    /*  36  0x24  */   eSIPP_STATUS_NOT_SET,
    /*  37  0x25  */   eSIPP_STATUS_NOT_HOOKED,
    /*  38  0x26  */   eSIPP_STATUS_COMPLETE,
    /*  39  0x27  */   eSIPP_STATUS_INVALID_NODE,
    /*  40  0x28  */   eSIPP_STATUS_DUPLICATE_NODE,
    /*  41  0x29  */   eSIPP_STATUS_HARDWARE_NOT_FOUND,
    /*  42  0x2A  */   eSIPP_STATUS_ILLEGAL_OPERATION,
    /*  43  0x2B  */   eSIPP_STATUS_INCOMPATIBLE_FORMATS,
    /*  44  0x2C  */   eSIPP_STATUS_INVALID_DEVICE,
    /*  45  0x2D  */   eSIPP_STATUS_INVALID_EDGE,
    /*  46  0x2E  */   eSIPP_STATUS_INVALID_NUMBER,
    /*  47  0x2F  */   eSIPP_STATUS_INVALID_STATE,
    /*  48  0x30  */   eSIPP_STATUS_INVALID_TYPE,
    /*  49  0x31  */   eSIPP_STATUS_STOPPED,
    /*  50  0x32  */   eSIPP_STATUS_SUSPENDED,
    /*  51  0x33  */   eSIPP_STATUS_TERMINATED,
    /* Last Entry */   eSIPP_STATUS_CODE_LAST = eSIPP_STATUS_TERMINATED

} eSIPP_STATUS;

typedef enum
{
    /* (0x0) Pipeline events */
    eSIPP_PIPELINE_FINALISED = 0x0, // Pipeline has been finalised
    eSIPP_PIPELINE_RESCHEDULED,     // Pipeline rescheduling complete
    eSIPP_PIPELINE_FRAME_DONE,      // Frame complete event for pipeline
    eSIPP_PIPELINE_ITERS_DONE,      // Iterations complete event for pipeline
    eSIPP_PIPELINE_SYNC_OP_DONE,    // Internal event passed to access scheduler to trigger it to update pipeline status
    eSIPP_PIPELINE_STARTED          // Pipeline has been internally scheduled and is to commence operation

} eSIPP_PIPELINE_EVENT;

typedef void SIPP_PIPELINE_EVENT_DATA;

////////////////////////////////////////////////////////
// SIPP frame store config struct - when filters are
// directly sourcing from or writing to full frame stores
typedef struct
{
    uint32_t baseAddress;
    uint32_t lineStride;       // lineStride as well as line width for partial line mode
    uint32_t lineWidth;        // Polyfir needs to know this of-course
    uint32_t format;
    uint32_t numLines;         // Adding this for plane stride calculation in case app
                          // does not wish to process full frame store - otherwise
                          // consuming/producing filter's frame height could be used

    // Not adding number of planes as assume that will match the
    // consuming/producing filter for a source/sink frame store
} SippFrCfg, * pSippFrCfg;

#endif /* _SIPP_DEFS_H_ */
