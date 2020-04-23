///
/// @file
/// @copyright All code copyright Movidius Ltd 2019, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     This file lists the configuration structures for SIPP SW filters.
///


/**
 * Parameter structure of the box filter.
 * 
 * This filter applies the box filter to the source image using the specified
 * structuring element.
 *
 * Identifier: svuBoxFilter
 */
typedef enum {
    FMT_U8,
    FMT_U16,
    FMT_U32,
    FMT_F16,
    FMT_F32,
}boxDataFmt;

typedef struct
{
    boxDataFmt dataFormat;
    UInt32     filterSizeH;
    UInt32     filterSizeV;
    UInt32     normalize;
}
BoxFilterParam;


/**
 * Parameter structure of the box 11x11 filter.
 * 
 * This filter applies the box filter to the source image using the specified
 * structuring element.
 *
 * Identifier: svuBoxFilter11x11
 * Group: CV
 * Inputs:
 *      - Datatypes: UInt8
 *      - Kernels: 11x11
 * Output datatype: UInt8
 */
typedef struct
{
    UInt32 normalize; // 1 to normalize to kernel size; 0 otherwise.
} BoxFilter11x11Param;


/**
 * Parameter structure of the box 13x13 filter.
 * 
 * This filter applies the box filter to the source image using the specified
 * structuring element.
 *
 * Identifier: svuBoxFilter13x13
 * Group: CV
 * Inputs:
 *      - Datatypes: UInt8
 *      - Kernels: 13x13
 * Output datatype: UInt8
 */
typedef struct
{
    UInt32 normalize; // 1 to normalize to kernel size; 0 otherwise.
} BoxFilter13x13Param;


/**
 * Parameter structure of the box 15x15 filter.
 * 
 * This filter applies the box filter to the source image using the specified
 * structuring element.
 *
 * Identifier: svuBoxFilter15x15
 * Group: CV
 * Inputs:
 *      - Datatypes: UInt8
 *      - Kernels: 15x15
 * Output datatype: UInt8
 */
typedef struct
{
    UInt32 normalize; // 1 to normalize to kernel size; 0 otherwise.
} BoxFilter15x15Param;


/**
 * Parameter structure of the box 3x3 filter.
 * 
 * This filter applies the box filter to the source image using the specified
 * structuring element.
 *
 * Identifier: svuBoxFilter3x3
 * Group: CV
 * Inputs:
 *      - Datatypes: UInt8
 *      - Kernels: 3x3
 * Output datatype: UInt8
 */
typedef struct
{
    UInt32 normalize; // 1 to normalize to kernel size; 0 otherwise.
} BoxFilter3x3Param;


/**
 * Parameter structure of the box 5x5 filter.
 * 
 * This filter applies the box filter to the source image using the specified
 * structuring element.
 *
 * Identifier: svuBoxFilter5x5
 * Group: CV
 * Inputs:
 *      - Datatypes: UInt8
 *      - Kernels: 5x5
 * Output datatype: UInt8
 */
typedef struct
{
    UInt32 normalize; // 1 to normalize to kernel size; 0 otherwise.
} BoxFilter5x5Param;


/**
 * Parameter structure of the box 7x7 filter.
 * 
 * This filter applies the box filter to the source image using the specified
 * structuring element.
 *
 * Identifier: svuBoxFilter7x7
 * Group: CV
 * Inputs:
 *      - Datatypes: UInt8
 *      - Kernels: 7x7
 * Output datatype: UInt8
 */
typedef struct
{
    UInt32 normalize; // 1 to normalize to kernel size; 0 otherwise.
} BoxFilter7x7Param;


/**
 * Parameter structure of the box 9x9 filter.
 * 
 * This filter applies the box filter to the source image using the specified
 * structuring element.
 *
 * Identifier: svuBoxFilter9x9
 * Group: CV
 * Inputs:
 *      - Datatypes: UInt8
 *      - Kernels: 9x9
 * Output datatype: UInt8
 */
typedef struct
{
    UInt32 normalize; // 1 to normalize to kernel size; 0 otherwise.
} BoxFilter9x9Param;


/**
 * Parameter structure of the channelExtract filter.
 *
 * Identifier: svuChannelExtract
 */
typedef struct
{
    UInt32 plane;
}
ChannelExtractParam;


/**
 * Parameter structure of the convolution 1x5 filter.
 *
 * This filter performs a convolution on the input image using the given 1x5
 * matrix.
 *
 * Identifier: svuConv1x5
 */
typedef struct
{
    UInt16 cMat[5];
}
Conv1x5Param;

/**
 * Parameter structure of the convolution 1x5Fp16ToFp16 filter.
 *
 * This filter performs a convolution on the input image using the given 1x5
 * matrix.
 *
 * Identifier: svuConv1x5Fp16ToFp16
 */
typedef struct
{

    UInt16 cMat[5];
}
Conv1x5Fp16ToFp16Param;

/**
 * Parameter structure of the convolution 1x7Fp16ToFp16 filter.
 *
 * This filter performs a convolution on the input image using the given 1x7
 * matrix.
 *
 * Identifier: svuConv1x7Fp16ToFp16
 */
typedef struct
{

    UInt16 cMat[7];
}
Conv1x7Fp16ToFp16Param;

/**
 * Parameter structure of the convolution 3x3 filter.
 *
 * This filter performs a convolution on the input image using the given 3x3
 * matrix.
 *
 * Identifier: svuConv3x3
 * Group: Arithmetic
 * Inputs:
 *      - Datatypes: UInt8
 *      - Kernels: 3x3
 * Output datatype: UInt8
 */
typedef struct
{
    //UInt16* cMat;
    UInt16 cMat[9];     // 3x3 convolution matrix.
                        // Default values:
                        // 0x2C00, 0x3000, 0x2C00,
                        // 0x3000, 0x3400, 0x3000,
                        // 0x2C00, 0x3000, 0x2C00
} Conv3x3Param;

/**
 * Parameter structure of the cConv3x3s2hhhh filter.
 *
 * This filter performs a convolution on the input image using the given 3x3
 * matrix.
 *
 * Identifier: svuConv3x3s2hhhh
 */
typedef struct
{
    UInt16 cMat[9];
}
Conv3x3s2hhhhParam;


/**
 * Parameter structure of the cConv3x3s3hhhh filter.
 *
 * This filter performs a convolution on the input image using the given 3x3
 * matrix.
 *
 * Identifier: svuConv3x3s3hhhh
 */
typedef struct
{
    UInt16 cMat[9];
}
Conv3x3s3hhhhParam;


/**
 * Parameter structure of the cConv3x3s4hhhh filter.
 *
 * This filter performs a convolution on the input image using the given 3x3
 * matrix.
 *
 * Identifier: svuConv3x3s4hhhh
 */
typedef struct
{
    UInt16 cMat[9];
}
Conv3x3s4hhhhParam;


/**
 * Parameter structure of the cConv3x3s8hhhh filter.
 *
 * This filter performs a convolution on the input image using the given 3x3
 * matrix.
 *
 * Identifier: svuConv3x3s8hhhh
 */
typedef struct
{
    UInt16 cMat[9];
}
Conv3x3s8hhhhParam;

/**
 * Parameter structure of the convolution 5x1 filter.
 *
 * This filter performs a convolution on the input image using the given 5x1
 * matrix.
 *
 * Identifier: svuConv5x1
 */
typedef struct
{
    UInt16 cMat[5]; 
}
Conv5x1Param;

/**
 * Parameter structure of the convolution 5x1 Fp16ToFp16 filter.
 *
 * This filter performs a convolution on the input image using the given 5x1
 * matrix.
 *
 * Identifier: svuConv5x1Fp16ToFp16 
  */
typedef struct
{

    UInt16 cMat[5];
}
Conv5x1Fp16ToFp16Param;

/**
 * Parameter structure of the convolution 5x5 filter.
 *
 * This filter performs a convolution on the input image using the given 5x5
 * matrix.
 *
 * Identifier: svuConv5x5
 */
typedef struct
{
    UInt16 cMat[25];
}
Conv5x5Param;

/**
 * Parameter structure of the Conv5x5s2hhhh filter.
 *
 * This filter performs a convolution on the input image using the given 5x5
 * matrix.
 *
 * Identifier: svuConv5x5s2hhhh
 */
typedef struct
{
    UInt16 cMat[25];
}
Conv5x5s2hhhhParam;


/**
 * Parameter structure of the Conv5x5s3hhhh filter.
 *
 * This filter performs a convolution on the input image using the given 5x5
 * matrix.
 *
 * Identifier: svuConv5x5s3hhhh
 */
typedef struct
{
    UInt16 cMat[25];
}
Conv5x5s3hhhhParam;


/**
 * Parameter structure of the Conv5x5s4hhhh filter.
 *
 * This filter performs a convolution on the input image using the given 5x5
 * matrix.
 *
 * Identifier: svuConv5x5s4hhhh
 */
typedef struct
{
    UInt16 cMat[25];
}
Conv5x5s4hhhhParam;


/**
 * Parameter structure of the Conv5x5s8hhhh filter.
 *
 * This filter performs a convolution on the input image using the given 5x5
 * matrix.
 *
 * Identifier: svuConv5x5s8hhhh
 */
typedef struct
{
    UInt16 cMat[25];
}
Conv5x5s8hhhhParam;


/**
 * Parameter structure of the convolution 7x1 filter.
 *
 * This filter performs a convolution on the input image using the given 7x1
 * matrix.
 *
 * Identifier: svuConv7x1
 */
typedef struct
{
     UInt16 cMat[7];
}
Conv7x1Param;


/**
 * Parameter structure of the convolution 7x7 fp16 to u8 filter.
 *
 * This filter performs a convolution on the input image using the given 7x7
 * matrix.
 *
 * Identifier: svuConv7x7Fp16ToU8
 */
typedef struct
{
    UInt16 cMat[49];
}
Conv7x7ParamFp16ToU8;

/**
 * Parameter structure of the convolution 7x7 s2hhhh filter.
 *
 * This filter performs a convolution on the input image using the given 7x7
 * matrix.
 *
 * Identifier: svuConv7x7s2hhhh
 */
typedef struct
{
    UInt16 cMat[49];
}
Conv7x7s2hhhhParam;

/**
 * Parameter structure of the convolution 7x7 s4hhhh filter.
 *
 * This filter performs a convolution on the input image using the given 7x7
 * matrix.
 *
 * Identifier: svuConv7x7s4hhhh
 */
typedef struct
{
    UInt16 cMat[49];
}
Conv7x7s4hhhhParam;

/**
 * Parameter structure of the convolution 7x7 s8hhhh filter.
 *
 * This filter performs a convolution on the input image using the given 7x7
 * matrix.
 *
 * Identifier: svuConv7x7s8hhhh
 */
typedef struct
{
    UInt16 cMat[49];
}
Conv7x7s8hhhhParam;


/**
 * Parameter structure of the convSeparable3x3 filter.
 *
 * Identifier: svuConvSeparable3x3
 */
typedef struct
{
    float cMat[2];
}
ConvSeparable3x3Param;

/**

/**
 * Parameter structure of the convSeparable3x3Fp16ToFp16 filter.
 *
 * Identifier: svuConvSeparable3x3Fp16ToFp16
 */
typedef struct
{

    UInt16 cMat[2]; 
}
ConvSeparable3x3Fp16ToFp16Param;

/**
 * Parameter structure of the convSeparable5x5 filter.
 *
 * Identifier: svuConvSeparable5x5
 */
typedef struct
{
    float cMat[3];
}
ConvSeparable5x5Param;

/**
 * Parameter structure of the convSeparable5x5Fp16ToFp16 filter.
 *
 * Identifier: svuConvSeparable5x5Fp16ToFp16
 */
typedef struct
{
    UInt16 cMat[3]; 
}
ConvSeparable5x5Fp16ToFp16Param;

/**
 * Parameter structure of the convSeparable7x7 filter.
 *
 * Identifier: svuConvSeparable7x7
 */
typedef struct
{
    float cMat[4];
}
ConvSeparable7x7Param;

/**
 * Parameter structure of the convSeparable7x7Fp16ToFp16 filter.
 *
 * Identifier: svuConvSeparable7x7Fp16ToFp16
 */
typedef struct
{
    UInt16 cMat[4]; 
}
ConvSeparable7x7Fp16ToFp16Param;


/**
 * Parameter structure of the convSeparable9x9 filter.
 *
 * Identifier: svuConvSeparable9x9
 */
typedef struct
{
    float cMat[5];
}
ConvSeparable9x9Param;


/**
 * Parameter structure of the dilate7x7 filter.
 *
 * Identifier: svuDilate7x7
 */
typedef struct
{
    UInt32 dMat[14];
}
Dilate7x7Param;


/**
 * Parameter structure of the erode3x3 filter.
 *
 * Identifier: svuErode3x3
 */
typedef struct
{
    UInt32 eMat[3];
}
Erode3x3Param;


/**
 * Parameter structure of the erode5x5 filter.
 *
 * Identifier: svuErode5x5
 */
typedef struct
{
    UInt32 eMat[10];
}
Erode5x5Param;


/**
 * Parameter structure of the HistogramStat filter.
 *
 * Identifier: svuHistogramStat
 */
typedef struct
{
    UInt32 step;
}
HistogramStatParam;


/**
 * Parameter structure of the positionKernel filter.
 *
 * Identifier: svuPositionKernel
 */
typedef struct
{
    UInt8 maskAddr[80];
    UInt8 pixelValue;
    UInt32 pixelPosition;
    UInt8 status;
}
positionKernelParam;


/**
 * Parameter structure of the scale bilinear planar filter.
 *
 * Identifier: svuScaleBilinearPlanar
 */
typedef struct
{
    UInt32 nChan;
    UInt32 firstSlice;
} ScaleBilinearPlanarParams;


/**
 * Parameter structure of the threshold filter.
 *
 * Identifier: svuThreshold
 */
enum
{
    Thresh_To_Zero       = 0,
    Thresh_To_Zero_Inv   = 1,
    Thresh_To_Binary     = 2,
    Thresh_To_Binary_Inv = 3,
    Thresh_Trunc         = 4
};
typedef struct
{
    UInt8 thresholdValue;
    UInt32 threshType;
}
ThresholdParam;


/**
 * Parameter structure of the threshold binary range filter.
 *
 * Identifier: svuThresholdBinaryRange
 */
typedef struct
{
    UInt8 lowerValue;
    UInt8 upperValue;

}
ThresholdBinaryRangeParam;


/**
 * Parameter structure of the threshold binary u8 filter.
 *
 * Identifier: svuThresholdBinaryU8
 */
typedef struct
{
    UInt8 threshold;
}
ThresholdBinaryU8Param;