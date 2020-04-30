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
    svuMerge3p                            = BASE_SW_ID + 149,
    //svuConv1x5                            = BASE_SW_ID + 29,
    //svuConv1x5Fp16ToFp16                  = BASE_SW_ID + 30,
    //svuConv1x7Fp16ToFp16                  = BASE_SW_ID + 31,
    //svuConv3x3                            = BASE_SW_ID + 32,
    //svuConvolution3x3s2hhhh               = BASE_SW_ID + 33,
    //svuConvolution3x3s3hhhh               = BASE_SW_ID + 34,
    //svuConvolution3x3s4hhhh               = BASE_SW_ID + 35,
    //svuConvolution3x3s8hhhh               = BASE_SW_ID + 36,
    //svuConv5x1                            = BASE_SW_ID + 37,
    //svuConv5x1Fp16ToFp16                  = BASE_SW_ID + 38,
    //svuConv5x5                            = BASE_SW_ID + 39,
    //svuConvolution5x5s2hhhh               = BASE_SW_ID + 40,
    //svuConvolution5x5s3hhhh               = BASE_SW_ID + 41,
    //svuConvolution5x5s4hhhh               = BASE_SW_ID + 42,
    //svuConvolution5x5s8hhhh               = BASE_SW_ID + 43,
    //svuConv7x1                            = BASE_SW_ID + 44,
    //svuConv7x7Fp16ToU8                    = BASE_SW_ID + 45,
    //svuConvolution7x7s2hhhh               = BASE_SW_ID + 46,
    //svuConvolution7x7s4hhhh               = BASE_SW_ID + 47,
    //svuConvolution7x7s8hhhh               = BASE_SW_ID + 48,
    //svuConvSeparable3x3                   = BASE_SW_ID + 49,
    //svuConvSeparable3x3Fp16ToFp16         = BASE_SW_ID + 50,
    //svuConvSeparable5x5                   = BASE_SW_ID + 51,
    //svuConvSeparable5x5Fp16ToFp16         = BASE_SW_ID + 52,
    //svuConvSeparable7x7                   = BASE_SW_ID + 53,
    //svuConvSeparable7x7Fp16ToFp16         = BASE_SW_ID + 54,
    //svuConvSeparable9x9                   = BASE_SW_ID + 55,
    svucvtColorNV12toBGR                  = BASE_SW_ID + 29,//56,
    svucvtColorNV12toRGB                  = BASE_SW_ID + 30,//57,
    svucvtColorNV21toRGB                  = BASE_SW_ID + 31,//58,
    svuCvtColorRGBtoLuma                  = BASE_SW_ID + 32,//59,
    svuCvtColorRGBToYUV422                = BASE_SW_ID + 33,//60,
    svuDilate7x7                          = BASE_SW_ID + 34,//61,
    svuErode3x3                           = BASE_SW_ID + 35,//62,
    svuErode5x5                           = BASE_SW_ID + 36,//63,
    svuGauss                              = BASE_SW_ID + 37,//64,
    svuGaussHx2                           = BASE_SW_ID + 38,//65,
    svuGaussHx2_fp16                      = BASE_SW_ID + 39,//66,
    svuGaussVx2                           = BASE_SW_ID + 40,//67,
    svuGaussVx2_fp16                      = BASE_SW_ID + 41,//68,
    svuHistogramStat                      = BASE_SW_ID + 42,//69,
    svuIntegralImageSqSumU32M2            = BASE_SW_ID + 43,//70,
    svuIntegralImageSumU16U32             = BASE_SW_ID + 44,//71,
    svuLaplacian3x3                       = BASE_SW_ID + 45,//72,
    svuLaplacian5x5                       = BASE_SW_ID + 46,//73,
    svuLaplacian7x7                       = BASE_SW_ID + 47,//74,
    svuMedianFilter11x11                  = BASE_SW_ID + 48,//75,
    svuMedianFilter13x13                  = BASE_SW_ID + 49,//76,
    svuMedianFilter15x15                  = BASE_SW_ID + 50,//77,
    svuMedianFilter3x3                    = BASE_SW_ID + 51,//78,
    svuMedianFilter5x5                    = BASE_SW_ID + 52,//79,
    svuMedianFilter7x7                    = BASE_SW_ID + 53,//80,
    svuMedianFilter9x9                    = BASE_SW_ID + 54,//81,
    svuPositionKernel                     = BASE_SW_ID + 55,//82,
    svuPyrDown                            = BASE_SW_ID + 56,//83,
    svuSAD11x11                           = BASE_SW_ID + 57,//84,
    svuSAD5x5                             = BASE_SW_ID + 58,//85,
    svuScale05BilinHV_Fp16U8              = BASE_SW_ID + 59,//86,
    svuScale05BilinHVFp16                 = BASE_SW_ID + 60,//87,
    svuScale2xBilinHV_Fp16U8_phase025_075 = BASE_SW_ID + 61,//88,
    svuScale2xBilinHV_U8ToU8_phase025_075 = BASE_SW_ID + 62,//89,
    svuScaleBilinearPlanar                = BASE_SW_ID + 63,//90,
    svuScl05Lanc6                         = BASE_SW_ID + 64,//91,
    svuScharr_fp16                        = BASE_SW_ID + 65,//92,
    svuSobel                              = BASE_SW_ID + 66,//93,
    svuSSD11x11                           = BASE_SW_ID + 67,//94,
    svuSSD5x5                             = BASE_SW_ID + 68,//95,
    svuSSD7x7U8ToU32                      = BASE_SW_ID + 69,//96,
    svuSsdPointLine7x7U8U32               = BASE_SW_ID + 70,//97,
    svuThreshold                          = BASE_SW_ID + 71,//98,
    svuThresholdBinaryRange               = BASE_SW_ID + 72,//99,
    svuThresholdBinaryU8                  = BASE_SW_ID + 73//100
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
