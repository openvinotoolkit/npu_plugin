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

//===================================================================
//MinMax params
/// @sf_definition minmaxma2x9x Minimum Maximum
/// @sf_description The MinMax filter finds the maximum and minimum points in scale space
/// @sf_group SIPP_ma2x9x_Hardware_Filters
/// @sf_myriadtarget ma2x9x
/// @sf_outdatatype fp16
/// @sf_type hw
/// @sf_preserve
/// @sf_function CV_MIN_MAX_ID
/// @sf_inputs
///     - datatypes: UInt16, fp16; kernels: 3x3
/// @{
/// @brief Parameter structure of the @ref minmax filter.
typedef struct {
    UInt16 scaling;            // half and double rate plane dimensions
    UInt16 fp16_mode;          // Input is in fp16 (16 bit)
    UInt16 rounding_en;        // Enable Rounding
    UInt16 ext_dense;          // extrema dense output
    UInt16 status_dense;       // status dense output
    UInt16 polarity;           // changes the polarity of extrema output bit[15]
    UInt16 numScoreThr;        // enable X score thresholds per Width and X per Height, or X*X per image
                               // Value grater than 4 will be treated as 4
    UInt16 noMinima;           // Minima disable, 1-minima is not reported, 0-minima is reported
    UInt16 max_extrema;        // maximum extrema
    /// @sf_pfdesc configuration bit field(see CV_MIN_MAX_THRESHOLD_ADR)
    UInt16 ext_threshold;
    /// @sf_pfdesc configuration bit field(see CV_MIN_MAX_SC_TH_LUT_REQ_ADR)
    UInt16 scoreThr;         // Score Threshold Lut
    /// @sf_pfdesc configuration bit field(see CV_MIN_MAX_ENABLE_ADR)
    UInt16 plane0_en;          // plane 0 enable
    UInt16 plane1_en;          // plane 1 enable
    UInt16 plane2_en;          // plane 2 enable
    UInt16 outStats_en;        // output stats enable
} MinMaxParam;
/// @}

//===================================================================
//Stereo Disparity params
/// @sf_definition stereodisparityma2x9x Stereo Disparity
/// @sf_description The Stereo Disparity filter extracts depth information based on epipolar differences
/// @sf_group SIPP_ma2x9x_Hardware_Filters
/// @sf_myriadtarget ma2x9x
/// @sf_outdatatype UInt8
/// @sf_type hw
/// @sf_preserve
/// @sf_function CV_STEREO_ID
/// @sf_inputs
///     - datatypes: UInt8; kernels: 5x5, 7x7, 7x9
/// @{

/// @brief Parameter structure of the @ref stereodisparity filter.
typedef struct {
/// @sf_pfprivate yes
    /// @sf_pfdesc configuration bit field(see CV_STEREO_CFG_ADR)
    UInt8 inputMode;       // Input mode
    UInt8 inputType;       // Image in mode (U8 and U10 pixel intensity)
    UInt8 dsp_sz;          // Disparity search size (0 = 64disp, 1 = 96disp)
    UInt8 search_dir;      // Cost Matchinh Search Direction (0 = L2R, 1 = R2L)
    UInt8 data_flow;       // Stereo data flow control (0 = Top, 1 = Full)
    UInt8 lrc_en;          // LR Check enable
    UInt8 cost_dump_en;    // Cost Map Dump enable
    UInt8 cost_dump_mode;  // Dump mode (0 = LA cost volume, 1 = SGBM cost volume)
    UInt8 wta_dump_en;     // WTA(SGBM) Dump enable
    UInt8 cme;             // companding enable
    UInt8 localAggr_mode;  // LA mode
    UInt8 dsp_out_mode;    // Output Disparity Map Format (0 = U(11,5), 1 = FP16)
    UInt8 axi_mode;        // AXI Input LLB mode
    /// @sf_pfdesc configuration bit field(see CV_STEREO_PARAMS_ADR)
    UInt8 invalid_dsp;     // Value to define the invalid disparity (defaults to 0)
    UInt8 invalid_thr;     // LRC Conditional Disparity Invalidity Threshold value
    UInt8 invalid_rep;     // Enable replace of invalid disparity with computed spatial kernel average
    UInt8 div_factor;      // SGBM Aggregation division factor
    UInt8 ratio_thr;       // SGBM WTA RATIO invalidation threshold
    /// @sf_pfdesc configuration bit field(see CV_STEREO_PARAMS2_ADR)
    UInt8 out_diff_thr;    // Kernel radius difference threshold
    UInt8 out_census_thr;  // Census Threshold
    UInt8 out_remove_thr;  // Outliers Remove Threshold

    UInt32 p1H;
    UInt32 p2H;
    UInt32 p1V;
    UInt32 p2V;

    UInt8  lut[256*4];

} StereoParam;

//===================================================================
//Warp params

typedef struct
{
    uint16_t numLines;              // The number of lines in the CB
    uint32_t memBase;               // The base of the CB
    uint32_t lineStride;            // Stride to next CB line
    int32_t cBufTop;               // Top line in CB
    int16_t CBL;                   // Location of top line
    int32_t cBufTopInit;           // Top line in CB @ start of frame
    int16_t CBLInit;               // Location of top line @ start of frame

} WarpCB, * pWarpCB;

// Descriptor word 0
typedef struct __attribute__ ((packed))
{
    uint64_t linkAddr          :32;  // Link address to next descriptor - bottom 4 bits unused
    uint64_t tileStart         :32;

} WarpDescWord0;

// Descriptor word 1
typedef struct __attribute__ ((packed))
{
    uint64_t intMeshStartXY :32;
    uint64_t fracMeshStartX :20;
    uint64_t mesh           :12;

} WarpDescWord1;

// Descriptor word 2
typedef struct __attribute__ ((packed))
{
    uint64_t fracMeshStartY :20;
    uint64_t gRecipX        :20;
    uint64_t gRecipY        :20;
    uint64_t reserved       :4;

} WarpDescWord2;

// Descriptor word 3
typedef struct __attribute__ ((packed))
{
    uint64_t pfbc           :20;
    uint64_t edgeColour     :17;
    uint64_t pixFormats     :17;
    uint64_t reserved       :6;
    uint64_t intClear       :4;

} WarpDescWord3;

// Descriptor word 4
typedef struct __attribute__ ((packed))
{
    uint64_t cbMemSetupA0    :28;
    uint64_t cbMemSetupA1    :28;
    uint64_t reserved        :4;
    uint64_t intEnable       :4;

} WarpDescWord4;

// Descriptor word 5
typedef struct __attribute__ ((packed))
{
    uint64_t cbMemSetupA2    :28;
    uint64_t cbMemSetupB0    :12;
    uint64_t cbMemSetupB1    :12;
    uint64_t cbMemSetupB2    :12;
//    uint64_t reserved        :3;

} WarpDescWord5;

// Descriptor word 6
typedef struct __attribute__ ((packed))
{
    uint64_t memSetup0       :32;
    uint64_t memSetup1       :32;

} WarpDescWord6;

// Descriptor word 7
typedef struct __attribute__ ((packed))
{
    uint64_t memSetup2       :32;
    uint64_t memBase0        :32;

} WarpDescWord7;

// Descriptor word 8
typedef struct __attribute__ ((packed))
{
    uint64_t memBase1        :32;
    uint64_t memBase2        :32;

} WarpDescWord8;

// Descriptor word 9
typedef struct __attribute__ ((packed))
{
    uint64_t outFrameLimit   :32;
    uint64_t inFrameLimitX   :32;

} WarpDescWord9;

// Descriptor word 10
typedef struct __attribute__ ((packed))
{
    uint64_t inFrameLimitY   :32;
    uint64_t meshLimit       :32;

} WarpDescWord10;

// Descriptor word 11
typedef struct __attribute__ ((packed))
{
    uint64_t mat0            :32;
    uint64_t mat1            :32;

} WarpDescWord11;

// Descriptor word 12
typedef struct __attribute__ ((packed))
{
    uint64_t mat2            :32;
    uint64_t mat3            :32;

} WarpDescWord12;

// Descriptor word 13
typedef struct __attribute__ ((packed))
{
    uint64_t mat4            :32;
    uint64_t mat5            :32;

} WarpDescWord13;

// Descriptor word 14
typedef struct __attribute__ ((packed))
{
    uint64_t mat6            :32;
    uint64_t mat7            :32;

} WarpDescWord14;

// Descriptor word 15
typedef struct __attribute__ ((packed))
{
    uint64_t mat8            :32;
    uint64_t mode            :13;
    uint64_t reserved        :19;

} WarpDescWord15;

// Descriptor word 16
typedef struct __attribute__ ((packed))
{
    uint64_t runs            :29;
    uint64_t start           :6;
    uint64_t reserved        :29;

} WarpDescWord16;

// Descriptor word 17
typedef struct __attribute__ ((packed))
{
    uint64_t reserved        :64;

} WarpDescWord17;

typedef struct
{
    WarpDescWord0  word0;
    WarpDescWord1  word1;
    WarpDescWord2  word2;
    WarpDescWord3  word3;
    WarpDescWord4  word4;
    WarpDescWord5  word5;
    WarpDescWord6  word6;
    WarpDescWord7  word7;
    WarpDescWord8  word8;
    WarpDescWord9  word9;
    WarpDescWord10 word10;
    WarpDescWord11 word11;
    WarpDescWord12 word12;
    WarpDescWord13 word13;
    WarpDescWord14 word14;
    WarpDescWord15 word15;
    WarpDescWord16 word16;
    WarpDescWord17 word17;

} WarpDesc, * pWarpDesc;

typedef enum
{
    SIPP_CV_WARP_SINGLE_CTX = 1,
    SIPP_CV_WARP_DOUBLE_CTX,
    SIPP_CV_WARP_TRIPLE_CTX

} SIPP_CV_WARP_OP_MODE;

typedef enum
{
    SIPP_CV_WARP_OBUF_SOLO = 0,
    SIPP_CV_WARP_OBUF_MASTER,
    SIPP_CV_WARP_OBUF_SLAVE

} SIPP_CV_WARP_OBUF_TYPE;


typedef struct
{
    UInt8  meshType;           // 0 : Sparse mesh, 1: Pre-expanded mesh
    UInt8  meshFormat;         // 0 : Mixed point 16 bit, 1 : FP32 - must be 0 when meshType is pre-expanded
    UInt8  meshRelative;       // 1 : MPs are relative to the output pixel location, 0 otherwise - must be 0 when meshType = sparse
    UInt8  meshDecPosn;        // The decimal position of the mixed point mesh points, counting from the LSB - 0 means fully integer
    UInt8  meshBypass;         // Set to generate a bypass mesh (NOTE: Must have PREEXPANDED_MODE = 1, RELATIVE = 1 and DEC_POSN = 4'b0)
    UInt32 meshBase;           // Base address of mesh
    UInt32 meshStride;         // Stride of mesh

    UInt32 inBufBase;          // Base address of input buffer - used if not using CB
    UInt32 inBufStride;        // Stride of input buffer in bytes - used if not using CB

    UInt32 outBufBase;         // Base address of output buffer - used if not using CB
    UInt32 outBufStride;       // Stride of output buffer in bytes - used if not using CB

    UInt8  filterMode;         // 0 : bilinear, 1 : bicubic, 2 : Bypass
    UInt8  bypassTransform;    // 0 : FALSE, 1 : TRUE

    UInt16 inputX;             // X size of input image (since WARP may have no parents can't guaranetee to get this
                               // via a filter link)
    UInt16 inputY;             // Y size of input image
    UInt16 startX;             // X co-ord of start location within the output image
    UInt16 startY;             // Y co-ord of start location within the output image
    UInt16 endX;               // X co-ord of end location within the output image
    UInt16 endY;               // Y co-ord of end location within the output image

    UInt16 meshWidth;          // Width of the mesh
    UInt16 meshHeight;         // Height of the mesh
    UInt16 meshOutSpan;        // Span of the mesh - states the output width the full mesh is intended to span
                               // In most cases this will equal the output width, even if resizing
                               // If striping, this should be the width were this context handling the whole output
    UInt16 meshOutSpanHeight;  // Span of the mesh - states the output height the full mesh is intended to span

    UInt32 transformMatrix[9]; // 3x3 transform matrix (fp32)

    UInt8  edgeMode;           // 0: Pixel replication, 1: Edge colour
    UInt16 edgeColour;         // Value to use when edgeMode = Edge colour

    UInt16 inPixWidth;         // The width of the input pixels in bits (set to 0 for fp16)
    UInt16 outPixWidth;        // The width of the output pixels in bits (set to 0 for fp16)

    UInt16 maxMeshYPositive;   // Maximum positive mesh point Y offset
    UInt16 maxMeshYNegative;   // Maximum negative mesh point Y offset

    UInt8  hFlip;              // Warp is being setup for horizontal flip so allow speculative cache fetching to the left

    UInt8  pfbcReqMode;        // Set to zero to disable Speculative block requests; Set to 1 to fetch the current Superblk (2x2blks); set to 2 to fetch 2 Superblks; etc... up to max of 4
    UInt8  tileXPrefLog2;      // application preferred tile X dimension - log base 2; valid values: [3..7]; set to 0 to leave it up to the SIPP default handling
                               // for application preferred tile Y dimension either set the number of lines per iteration of the pipeline warp is part of (Scheduled RT) or the IRQ rate of the
                               // warp filters used (FREE RT)

    //
    pWarpDesc pDesc;

/// @sf_pfprivate yes
    UInt8  tileXLog2;          // tile X dimension - log base 2
/// @sf_pfprivate yes
    UInt8  tileYLog2;          // tile Y dimension - log base 2
    UInt8  numTilesRow;        // How many tiles in an row
    UInt32 gRecipX;
    UInt32 gRecipY;
    UInt16 meshStartIntX;      // X mesh start point integer part  - Will client ever set this?
    UInt32 meshStartFracX;     // X mesh start point fraction part
    UInt16 meshStartIntY;      // Y mesh start point integer part
    UInt32 meshStartFracY;     // Y mesh start point fraction part

    WarpCB * pInCB;            // Pointer to a CB struct (should be an array with an entry for every image)
    WarpCB * pOutCB;           // Pointer to a CB struct (should be an array with an entry for every image)

    Int16    runsComplete;
    UInt16   locMeshIntY;
    UInt32   locMeshFracY;
    UInt16   locStartY;
    UInt8    locSuperRun;
    UInt16   iBufLevel;          // Maybe iBufLevel does not matter - its just nextRunStart and inLineNum we care about
                                 // its only used for freeing
    UInt16   oBufLevel;          // Note this includes pending transactions

    UInt16   inLineNum;
    UInt16   firstLineStart;
    UInt16   nextRunStart;
    bool     bFreeMemOnRowDone;   // When set, free one tile row from the input buffer at the end of a tile row.

} warpCtx;

/// @sf_definition warpma2x9x Bicubic Warp
/// @sf_description Warp Filter
/// @sf_group SIPP_ma2x9x_Hardware_Filters
/// @sf_myriadtarget ma2x9x
/// @sf_outdatatype fp16 / UInt16, u14, 12, u10, uint8_t
/// @sf_type hw
/// @sf_preserve
/// @sf_function CV_WARP_X_ID
/// @sf_inputs
///     - datatypes: UInt16, fp16
/// @{

/// @brief Parameter structure of the @ref minmax filter.
typedef struct {

    warpCtx * pContext[SIPP_MAX_WARP_CTXS];

    // This variables should be common to all input contexts / planes - no need to allocate
    // multiple times.
    // Now we need some other type of mode here when we want a filter to execute on more
    // than one image in parallel such as when doing U and V but that can come later...
    SIPP_CV_WARP_OP_MODE   opMode;         // 0: Single context/plane 1: Multi Context / plane
    SIPP_CV_WARP_OBUF_TYPE oBufType;       // Signals if the warp output buffer is to be the master buffer in a set
                                           // This would apply for the output buffers for ALL contexts / planes
    SippFilters *           masterOBufFilt; // if oBufType == SIPP_CV_WARP_OBUF_SLAVE, this signals where the master
                                           // filter is
    SippFilters *           slaveOBufFilts[(SIPP_HWCV_NUM_WARP_UNITS - 0x1)];
                                           // if oBufType == SIPP_CV_WARP_OBUF_MASTER, this signals where the slave
                                           // filter(s) are
    uint8_t                     slaveOutContext;// Effectively states which output context this slave will be working on
                                           // In a striping scenario, all slaves will also be working on context zero
                                           // In a plane stacking scenario where filter feeds a circular buffer,
                                           // the filter needs this info in order to correctly place output into the
                                           // correct plane / context
/// @sf_pfprivate yes
    #ifdef SIPP_WARP_NO_CTX_LINK
    UInt8                currCtx;
    #endif
    UInt32               frmDim;             // see CV_MIN_MAX_FRM_DIM_ADR (Private)
    UInt8                readyFlags;
    bool                 bTransInProg;       // Note - this is not accurate for single ctx filters but it is used only in multi-ctx
    Int16                lastRowReport;
    UInt16               runsStarted;
    UInt16               totalRuns;
    bool                 bStackOutCtx;       // When stacking output planes - a filter is essentially covering more than one
                                             // context on input but should only have one output!

} WarpParam;
/// @}

typedef struct
{
    UInt8  meshType;           // 0 : Sparse mesh, 1: Pre-expanded mesh
    UInt8  meshFormat;         // 0 : Mixed point 16 bit, 1 : FP32 - must be 0 when meshType is pre-expanded
    UInt8  meshRelative;       // 1 : MPs are relative to the output pixel location, 0 otherwise - must be 0 when meshType = sparse
    UInt8  meshDecPosn;        // The decimal position of the mixed point mesh points, counting from the LSB - 0 means fully integer
    UInt8  meshBypass;         // Set to generate a bypass mesh (NOTE: Must have PREEXPANDED_MODE = 1, RELATIVE = 1 and DEC_POSN = 4'b0)
    UInt32 meshBase;           // Base address of mesh
    UInt32 meshStride;         // Stride of mesh

    UInt32 inBufBase;          // Base address of input buffer - used if not using CB
    UInt32 inBufStride;        // Stride of input buffer in bytes - used if not using CB

    UInt32 outBufBase;         // Base address of output buffer - used if not using CB
    UInt32 outBufStride;       // Stride of output buffer in bytes - used if not using CB

    UInt8  filterMode;         // 0 : bilinear, 1 : bicubic, 2 : Bypass
    UInt8  bypassTransform;    // 0 : FALSE, 1 : TRUE

    UInt16 inputX;             // X size of input image (since WARP may have no parents can't guaranetee to get this
                               // via a filter link)
    UInt16 inputY;             // Y size of input image
    UInt16 startX;             // X co-ord of start location within the output image
    UInt16 startY;             // Y co-ord of start location within the output image
    UInt16 endX;               // X co-ord of end location within the output image
    UInt16 endY;               // Y co-ord of end location within the output image

    UInt16 meshWidth;          // Width of the mesh
    UInt16 meshHeight;         // Height of the mesh
    UInt16 meshOutSpan;        // Span of the mesh - states the output width the full mesh is intended to span
                               // In most cases this will equal the output width, even if resizing
                               // If striping, this should be the width were this context handling the whole output
    UInt16 meshOutSpanHeight;  // Span of the mesh - states the output height the full mesh is intended to span

    UInt8  edgeMode;           // 0: Pixel replication, 1: Edge colour
    UInt16 edgeColour;         // Value to use when edgeMode = Edge colour

    UInt16 inPixWidth;         // The width of the input pixels in bits (set to 0 for fp16)
    UInt16 outPixWidth;        // The width of the output pixels in bits (set to 0 for fp16)

    UInt16 maxMeshYPositive;   // Maximum positive mesh point Y offset
    UInt16 maxMeshYNegative;   // Maximum negative mesh point Y offset

    UInt8  hFlip;              // Warp is being setup for horizontal flip so allow speculative cache fetching to the left

    UInt8  pfbcReqMode;        // Set to zero to disable Speculative block requests; Set to 1 to fetch the current Superblk (2x2blks); set to 2 to fetch 2 Superblks; etc... up to max of 4
    UInt8  tileXPrefLog2;      // application preferred tile X dimension - log base 2; valid values: [3..7]; set to 0 to leave it up to the SIPP default handling
                               // for application preferred tile Y dimension either set the number of lines per iteration of the pipeline warp is part of (Scheduled RT) or the IRQ rate of the
                               // warp filters used (FREE RT)

} ParamWarp;

#endif /* __SIPP_CV_DEFS_H__ */
