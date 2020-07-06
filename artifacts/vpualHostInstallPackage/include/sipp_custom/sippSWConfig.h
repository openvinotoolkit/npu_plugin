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
/// @file
/// @copyright All code copyright Movidius Ltd 2019, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     This file lists the configuration structures for SIPP SW filters.
///

#include <swcFrameTypes.h>

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
 * Parameter structure of the convolution 1x7 filter.
 *
 * This filter performs a convolution on the input image using the given 1x7
 * matrix.
 *
 * Identifier: svuConv1x7
 */
typedef struct
{
    UInt16 cMat[7];
}
Conv1x7Param;

/**
 * Parameter structure of the convolution 1x9 filter.
 *
 * This filter performs a convolution on the input image using the given 1x9
 * matrix.
 *
 * Identifier: svuConv1x9
 */
typedef struct
{
    UInt16 cMat[9];
}
Conv1x9Param;

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
 * Parameter structure of the convolution 7x7 filter.
 *
 * This filter performs a convolution on the input image using the given 7x7
 * matrix.
 *
 * Identifier: svuConv7x7
 */
typedef struct
{
    UInt16 cMat[49];
}
Conv7x7Param;

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
 * Parameter structure of the cvt color Luma N12 filter.
 *
 * Identifier: svucvtColorLumaNV12
 */typedef struct
{
    float coefsMat[9];
    float offset[3];
}
cvtColorLumaNV12Param;


/**
 * Parameter structure of the cvt color Chroma N12 filter.
 *
 * Identifier: svucvtColorChromaNV12
 */
typedef struct
{
    float coefsMat[9];
    float offset[3];
}
cvtColorChromaNV12Param;


/**
 * Parameter structure of the dilate7x7 filter.
 *
 * Identifier: svuDilate7x7
 */
typedef struct
{
    UInt8 dMat[49];
}
Dilate7x7Param;


/**
 * Parameter structure of the erode3x3 filter.
 *
 * Identifier: svuErode3x3
 */
typedef struct
{
    UInt8 eMat[9];
}
Erode3x3Param;


/**
 * Parameter structure of the erode5x5 filter.
 *
 * Identifier: svuErode5x5
 */
typedef struct
{
    UInt8 eMat[25];
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
 * Parameter structure of the scale bilinear filter.
 *
 * Identifier: svuScaleBilinear
 */
typedef struct
{
    UInt32 nChan;
    UInt32 firstSlice;
}
ScaleBilinearParams;


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


/**
 * Parameter structure of the Conv3x3s2xhhx filter.
 *
 * This filter performs a convolution on the input image using the given 3x3
 * matrix.
 *
 * Identifier: svuConvolution3x3s2xhhx
 */
typedef struct
{
    UInt16 cMat[9];
}
Conv3x3s2xhhxParam;


/**
 * Parameter structure of the scale Fp16 filter.
 *
 * Identifier: svuScaleFp16
 */
typedef struct
{
    UInt16 scale;
} ScaleFp16Param;


/**
 * Parameter structure of the Conv7x7s3hhhh filter.
 *
 * This filter performs a convolution on the input image using the given 7x7
 * matrix.
 *
 * Identifier: svuConvolution7x7s3hhhh
 */
typedef struct
{
    UInt16 cMat[49];
}
Conv7x7s3hhhhParam;


/**
 * Parameter structure of the convolution 3x3 Fp16ToFp16 filter.
 *
 * This filter performs a convolution on the input image using the given 3x3
 * matrix.
 *
 * Identifier: svuConv3x3Fp16ToFp16 
  */
typedef struct
{

    UInt16 cMat[9];
}
Conv3x3Fp16ToFp16Param;

/**
 * Parameter structure of the convolution 5x5 Fp16ToFp16 filter.
 *
 * This filter performs a convolution on the input image using the given 5x5
 * matrix.
 *
 * Identifier: svuConv5x5Fp16ToFp16 
  */
typedef struct
{

    UInt16 cMat[25];
}
Conv5x5Fp16ToFp16Param;

/**
 * Parameter structure of the convolution 7x1 Fp16ToFp16 filter.
 *
 * This filter performs a convolution on the input image using the given 7x1
 * matrix.
 *
 * Identifier: svuConv7x1Fp16ToFp16 
  */
typedef struct
{

    UInt16 cMat[7];
}
Conv7x1Fp16ToFp16Param;

/**
 * Parameter structure of the convolution 7x7 Fp16ToFp16 filter.
 *
 * This filter performs a convolution on the input image using the given 7x7
 * matrix.
 *
 * Identifier: svuConv7x7Fp16ToFp16 
  */
typedef struct
{

    UInt16 cMat[49];
}
Conv7x7ParamFp16ToFp16;


/**
 * Parameter structure of the convolution 1x15 filter.
 *
 * This filter performs a convolution on the input image using the given 1x15
 * matrix.
 *
 * Identifier: svuConv1x15
 */
typedef struct
{
    UInt16 cMat[15];
}
Conv1x15Param;


/**
 * Parameter structure of the convolution 9x1 filter.
 *
 * This filter performs a convolution on the input image using the given 9x1
 * matrix.
 *
 * Identifier: svuConv9x1
 */
typedef struct
{
    UInt16 cMat[9];
}
Conv9x1Param;


/**
 * Parameter structure of the convolution 15x1 filter.
 *
 * This filter performs a convolution on the input image using the given 15x1
 * matrix.
 *
 * Identifier: svuConv15x1
 */
typedef struct
{
    UInt16 cMat[15];
}
Conv15x1Param;


/**
 * Parameter structure of the dilate5x5 filter.
 *
 * This filter performs a dilate on the source image using
 * the specified structuring element.
 *
 * Identifier: svuDilate5x5
 */
typedef struct
{
    UInt8 dMat[25];
}
Dilate5x5Param;


/**
 * Parameter structure of the randNoiseFp16 filter.
 *
 * This filter generates random noise using high speed algorithm.
 *
 * Identifier: svuGenNoiseFp16
 */
typedef struct
{
    float strength;
}
RandNoiseFp16Param;



/**
 * Parameter structure of the erode7x7 filter.
 *
 * This filter applies the erode filter on the source image using the specified structuring element.
 *
 * Identifier: svuErode7x7
 */
typedef struct
{
    UInt8 dMat[49];
}
Erode7x7Param;


/**
 * Parameter structure of the bilinearInterpolation filter.
 *
 * This kernel does bilinear interpolation when there is a pattern between bilinear factors
 * if the horizontal scalefactor(srcWidth/destWidth) has one decimal point precision
 *
 * Identifier: svuBilinearInterpolation
 */
typedef struct
{
    uint16_t precalcPos[10];
    uint16_t lut[40];
    int stopW;
    int accesPatternStep;
}
BilinearInterpolationParam;


/**
 * Parameter structure of the minMaxPos filter.
 *
 * This filter computes the minimum and the maximum value of a given input line and their position.
 *
 * Identifier: svuMinMaxPos
 */
typedef struct
{
    UInt8 Mask[80];
}
MinMaxPosParam;


/**
 * Parameter structure of the MinMaxValue filter.
 *
 * This filter computes the minimum and the maximum value of a given input image.
 *
 * Identifier: svuMinMaxValue
 */
typedef struct
{
    UInt8 maskAddr[80];
}
minMaxValParam;

/**
 * Parameter structure of the Pad_kernelu8 filter.
 *
 *  This kernel calculates mean and standard deviation of an array of elements
 *
 * Identifier: svuPadKernel_u8
 */
enum
{
    Left         = 0,
    Right        = 1,
    LeftAndRight = 2,
};

enum
{
    AllZero    = 0,
    AllOne     = 1,
    Mirror     = 2,
    BlackPixel = 3,
    WhitePixel = 4,
    PixelValue = 5,
};

typedef struct
{
    UInt32 padSz;
    UInt32 padMode;
    UInt32 padType;
    UInt8  pixValue;
}
PadKernel_u8Param;

/**
 * Parameter structure of the Pad_kernelu16 filter.
 *
 *  This kernel calculates mean and standard deviation of an array of elements
 *
 * Identifier: svuPadKernel_u16
 */
enum
{
    Left_u16         = 0,
    Right_u16        = 1,
    LeftAndRight_u16 = 2,
};

enum
{
    AllZero_u16    = 0,
    AllOne_u16     = 1,
    Mirror_u16     = 2,
    BlackPixel_u16 = 3,
    WhitePixel_u16 = 4,
    PixelValue_u16 = 5,
};

typedef struct
{
    UInt32 padSz;
    UInt32 padMode;
    UInt32 padType;
    UInt16 pixValue;
}
PadKernel_u16Param;

/**
 * Parameter structure of the dilate3x3 filter.
 *
 * This filter performs a dilate on the source image using
 * the specified structuring element.
 *
 * Identifier: svuDilate3x3
 */
typedef struct
{
    UInt8 dMat[9];
}
Dilate3x3Param;


/**
 * Parameter structure of the calcEpipolarDistance filter.
 *
 * The filter finds edges in the input image and marks them
 * in the output map edges using the Canny algorithm.
 *
 * Identifier: svuCalcEpipolarDistance
 */
typedef struct
{
    UInt32 nPoints;
    float  RANSAC_dist_threshold;
    float  fm[9];
}
calcEpipolarDistanceParam;


/**
 * Parameter structure of the calcG filter.
 *
 * The filter finds edges in the input image and marks them
 * in the output map edges.
 *
 * Identifier: svuCalcG
 */
typedef struct
{
    UInt32 isz[2];
    UInt32 jsz[2];
    UInt32 minI[2];
    UInt32 minJ[2];
}
calcGParam;


/**
 * Parameter structure of the calcBxBy filter.
 *
 * The filter finds edges in the input image and marks them
 * in the output map edges.
 *
 * Identifier: svuCalcBxBy
 */
typedef struct
{
    UInt32 isz[2];
    UInt32 jsz[2];
    UInt32 minI[2];
    UInt32 minJ[2];
}
calcBxByParam;


/**
 * Parameter structure of the cvtColorChromaYUVToNV12 filter.
 *
 * This filter performs conversion from YUV image format to NV12
 * for the chroma part only. The luma part is identical between
 * these two formats and needs to be copied separately.
 *
 * Identifier: svuCvtColorChromaYUVToNV12
 */
typedef struct
{
    frameType inputFrameType;
    UInt8 needs2Parents;
}
CvtColorChromaYUVToNV12Param;


/**
 * Parameter structure of the hammingDistance filter.
 *
 * This HammingDistance kernel finds matches between two descriptors
 *
 * Identifier: svuHammingDistance
 */
typedef struct
{
    int descriptor_size;
}
HammingDistanceParam;

/**
 * Parameter structure of the convSeparable11x11 filter.
 *
 * This filter performs a separable convolution on the input image using the given 11x11
 * matrix.
 * Identifier: svuConvSeparable11x11
 */
typedef struct
{
    UInt32 cMat[6];
}
ConvSeparable11x11Param;



/**
 * Parameter structure of the convSeparable11x11Fp16ToFp16 filter.
 *
 *  This filter performs a separable convolution on the fp16 input image using the given 11x11
 * matrix.
 * Identifier: svuConvSeparable11x11Fp16ToFp16
 */
typedef struct
{
    UInt16 cMat[6];
}
ConvSeparable11x11Fp16ToFp16Param;


/**
 * Parameter structure of the convSeparable9x9Fp16ToFp16 filter.
 *
 * This filter performs a separable convolution on the fp16 input image using the given 9x9
 * matrix.
 * Identifier: svuConvSeparable9x9Fp16ToFp16
 */
typedef struct
{
    UInt16 cMat[5];
}
ConvSeparable9x9Fp16ToFp16Param;


/**
 * Parameter structure of the Conv11x11s3hhhh filter.
 *
 * This filter performs a convolution the fp16 input image using the given 11x11
 * matrix.
 *
 * Identifier: svuConv11x11s3hhhh
 */
typedef struct
{
    UInt16 cMat[121];
}
Conv11x11s3hhhhParam;


/**
 * Parameter structure of the Conv11x11s4hhhh filter.
 *
 * This filter performs a convolution on the fp16 input image using the given 11x11
 * matrix.
 *
 * Identifier: svuConv11x11s4hhhh
 */
typedef struct
{
    UInt16 cMat[121];
}
Conv11x11s4hhhhParam;


/**
 * Parameter structure of the Conv11x11s8hhhh filter.
 *
 * This filter performs a convolution on the fp16 input image using the given 11x11
 * matrix.
 *
 * Identifier: svuConv11x11s8hhhh
 */
typedef struct
{
    UInt16 cMat[121];
}
Conv11x11s8hhhhParam;

/**
 * Parameter structure of the convolution 11x11 filter.
 *
 * This filter performs a convolution on the input image using the given 11x11
 * matrix.
 *
 * Identifier: svuConv11x11
 */
typedef struct
{
    UInt16 cMat[121];
}
Conv11x11Param;


/**
 * Parameter structure of the Conv11x11s1hhhh filter.
 *
 * This filter performs a convolution on the input image using the given 11x11
 * matrix.
 *
 * Identifier: svuConv11x11s1hhhh
 */
typedef struct
{
    UInt16 cMat[121];
}
Conv11x11s1hhhhParam;


/**
 * Parameter structure of the Conv11x11s1xxhx filter.
 *
 * This filter performs a convolution on the input image using the given 11x11
 * matrix.
 *
 * Identifier: svuConv11x11s1xxhx
 */
typedef struct
{
    UInt8 cMat[121];
}
Conv11x11s1xxhxParam;


/**
 * Parameter structure of the Conv11x11s2hhhh filter.
 *
 * This filter performs a convolution on the fp16 input image using the given 11x11
 * matrix.
 *
 * Identifier: svuConv11x11s2hhhh
 */
typedef struct
{
    UInt16 cMat[121];
}
Conv11x11s2hhhhParam;


/**
 * Parameter structure of the Conv11x11s2xxhx filter.
 *
 * This filter performs a convolution on the input image using the given 11x11
 * matrix.
 *
 * Identifier: svuConv11x11s2xxhx
 */
typedef struct
{
    UInt8 cMat[121];
}
Conv11x11s2xxhxParam;


/**
 * Parameter structure of the Conv3x3fp32Scharr filter.
 *
 * This filter performs a convolution on the input image using the given 3x3
 * matrix using Scharr.
 *
 * Identifier: svuConv3x3fp32Scharr
 */
typedef struct
{
    float smooth_k[2];
    int height;
    int width;
}
Conv3x3fp32ScharrParam;

/**
 * Parameter structure of the convolution9x9 filter.
 *
 * This filter performs a convolution on the input image using the given 9x9 matrix.
 *
 * Identifier: svuConv9x9
 */
typedef struct
{
    UInt16 cMat[81];
}
Conv9x9Param;


/**
 * Parameter structure of the convolution9x9Fp16ToFp16 filter.
 *
 * This filter performs a convolution on the fp16 input image using the given 9x9 matrix.
 *
 * Identifier: svuConv9x9Fp16ToFp16
 */
typedef struct
{
    UInt16 cMat[81];
}
Conv9x9ParamFp16ToFp16;


/**
 * Parameter structure of the convolution9x9s2hhhh filter.
 *
 * This filter performs a convolution on the fp16 input image using the given 9x9 matrix.
 * Identifier: svuConvolution9x9s2hhhh
 */
typedef struct
{
    UInt16 cMat[81];
}
Conv9x9s2hhhhParam;


/**
 * Parameter structure of the convolution9x9s3hhhh filter.
 *
 * This filter performs a convolution on the fp16 input image using the given 9x9 matrix.
 *
 * Identifier: svuConvolution9x9s3hhhh
 */
typedef struct
{
    UInt16 cMat[81];
}
Conv9x9s3hhhhParam;


/**
 * Parameter structure of the convolution9x9s4hhhh filter.
 *
 * This filter performs a convolution on the fp16 input image using the given 9x9 matrix.
 *
 * Identifier: svuConvolution9x9s4hhhh
 */
typedef struct
{
    UInt16 cMat[81];
}
Conv9x9s4hhhhParam;

/**
 * Parameter structure of the convolution9x9s8hhhh filter.
 *
 * This filter performs a convolution on the fp16 input image using the given 9x9 matrix.
 *
 * Identifier: svuConvolution9x9s8hhhh
 */
typedef struct
{
    UInt16 cMat[81];
}
Conv9x9s8hhhhParam;


/**
 * Parameter structure of the bilateral5x5 filter.
 *
 * This kernels performs a bilateral filter on the input image of 5x5 dimensions.
 *
 * Identifier: svuBilateral5x5
 */
typedef struct
{
    UInt16 sigma[384];
}
Bilateral5x5Param;


/**
 * Parameter structure of the histogram filter.
 *
 * This filter computes a histogram on a given line to be applied to all lines of an image.
 *
 * Identifier: svuHistogram
 */
typedef struct
{
    UInt32 hist[256];
}
HistogramParam;


/**
 * Parameter structure of the equalizeHist filter.
 *
 * his filter makes an equalization through an image with a given histogram.
 *
 * Identifier: svuEqualizeHist
 */
typedef struct
{
    UInt32 cum_hist[256];
}
EqualizeHistParam;


/**
 * Parameter structure of the dilateGeneric filter.
 *
 * This filter performs a generic dilate on the input image using the kernel size given by the user.
 *
 * Identifier: svuDilateGeneric
 */
typedef struct
{
    UInt32 dMat[61];
    // UInt32 kernelSize;
}
DilateGenericParam;


/**
 * Parameter structure of the cvtInterleavedtoPlanar filter.
 *
 * This filter performs a conversion from a interleaved input image to a planar output image.
 *
 * Identifier: svuCvtInterleavedtoPlanar
 */
typedef struct
{
    UInt32 planes;
}
CvtInterleavedtoPlanarParam;