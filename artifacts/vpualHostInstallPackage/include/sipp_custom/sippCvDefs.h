///
/// @file      sippCvDefs.h
/// @copyright All code copyright Movidius Ltd 2019, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     SIPP HW filter configuration structures.
///

#ifndef __SIPP_CV_DEFS_H__
#define __SIPP_CV_DEFS_H__

#include "sippDefs.h"

//===================================================================
//Edge operator params
/// @sf_definition edgeoperatorma2x9x Edge operator
/// @sf_description Flexible 3x3 edge-detection operator suitable for implementation of e.g. Sobel filter.
/// @sf_group SIPP_ma2x9x_Hardware_Filters
/// @sf_myriadtarget ma2x9x
/// @sf_preserve imgSize, numPlains
/// @sf_outdatatype UInt8, UInt16
/// @sf_type hw
/// @sf_function CV_EDGE_OP_ID
/// @sf_inputs
///     - datatypes: UInt8; kernels: 3x3
/// @{

/// @brief Parameter structure of the @ref edgeoperator filter.
typedef struct {
/// @sf_pfprivate yes
    UInt32 frmDim;            // see SIPP_EDGE_OP_FRM_DIM_ADR (Private)
    /// @sf_pfdesc configuration bitfield(see SIPP_EDGE_OP_CFG_ADR)
    UInt32 in_mode;    // input mode
    UInt32 out_mode;   // output mode
    UInt32 theta_mode; // theta mode
    UInt32 theta_ovx;  // theta OpenVx mode
    UInt32 pm;         // plane multiple out_mode
    UInt32 magn_scf;   // magnitude scale factor
     /// @sf_pfdesc Edge operator X coefficients(see SIPP_EDGE_OP_XCOEFF_ADR)
    UInt32 xCoeffA;
    UInt32 xCoeffB;
    UInt32 xCoeffC;
    UInt32 xCoeffD;
    UInt32 xCoeffE;
    UInt32 xCoeffF;
    /// @sf_pfdesc Edge operator Y coefficients(see SIPP_EDGE_OP_YCOEFF_ADR)
    UInt32 yCoeffA;
    UInt32 yCoeffB;
    UInt32 yCoeffC;
    UInt32 yCoeffD;
    UInt32 yCoeffE;
    UInt32 yCoeffF;

    SippFrCfg  inFrCfg;
    #ifdef SIPP_SUPPORT_FF_OBUF
    SippFrCfg  outFrCfg;
    #endif

} EdgeParam;
/// @}

//===================================================================
//Harris corner detect params
/// @sf_definition harriscornersma2x9x Harris Corner Detector
/// @sf_description The Harris corners filter performs corner detection on U8F image data.
/// @sf_group SIPP_ma2x9x_Hardware_Filters
/// @sf_myriadtarget ma2x9x
/// @sf_outdatatype half,fp16,fp32,float
/// @sf_type hw
/// @sf_preserve numPlanes, imgSize
/// @sf_function CV_HARRIS_ID
/// @sf_inputs
///     - datatypes: UInt8; kernels: 5x5, 7x7, 9x9
/// @{

/// @brief Parameter structure of the @ref harriscorners filter.
typedef struct {
/// @sf_pfprivate yes
    UInt32 frmDim;             // see SIPP_HARRIS_FRM_DIM_ADR (Private)
    /// @sf_pfdesc configuration bit field(see SIPP_HARRIS_CFG_ADR)
    UInt32 kernel_sz;          // kernel size
    UInt32 out_sel;            // select output
    UInt32 no_sobel;           // bypass Sobel operator
    UInt32 shi_tomasi;         // Shi-Tomasi score output
    UInt32 exp_sub;            // exponent subtrahend
    UInt32 pm;                 // plane multiple
    /// @sf_pfdesc value that changes the response of the edges(FP32)(see SIPP_HARRIS_K_ADR)
    float  kValue;             // see SIPP_HARRIS_K_ADR

    SippFrCfg  inFrCfg;
    #ifdef SIPP_SUPPORT_FF_OBUF
    SippFrCfg  outFrCfg;
    #endif

} HarrisParam;
/// @}

#endif /* __SIPP_CV_DEFS_H__ */
