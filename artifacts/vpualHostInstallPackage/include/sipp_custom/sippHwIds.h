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
/// @copyright All code copyright Movidius Ltd 2016, all rights reserved.
///            For License Warranty see: common/license.txt
///
/// @brief     HW filter related macros
///

#ifndef _SIPP_HW_IDS_2X9X_H_
#define _SIPP_HW_IDS_2X9X_H_

#ifdef SIPP_PC
#include "registersMyriad.h"
#endif

///////////////////////////////////////////////////////
// SIPP Filter/buffer IDs
// Following are filter and input/output buffer IDs

#define SIPP_FREE_RT_ID     -1    /* Convention to mark the FreeRT */

///////////////////////////////////////////////////////
// RAW processing
#define SIPP_SIGMA_ID        0   // Sigma denoise
#define SIPP_LSC_ID          1   // Lens shading Correction
#define SIPP_RAW_ID          2   // RAW filter
#define SIPP_LCA_ID          3   // Lateral Chromatic Aberration
#define SIPP_DBYR_ID         4   // Debayer

///////////////////////////////////////////////////////
// Luma processing
#define SIPP_LUMA_ID         5   // Luma denoise
#define SIPP_DOG_ID          6   // Difference of Gaussians
#define SIPP_SHARPEN_ID     12   // Sharpening
#define SIPP_LUMAFUS_ID     16   // Luma Fusion
#define SIPP_TONEMAP_ID     17   // Tone mapping

///////////////////////////////////////////////////////
// Chroma processing
#define SIPP_CGEN_ID         7   // Generate chroma filter
#define SIPP_MED_ID          8   // Median
#define SIPP_CHROMA_ID       9   // Chroma denoise
#define SIPP_CHRFUS_ID      18   // Chroma Fusion

///////////////////////////////////////////////////////
// RGB processing
#define SIPP_CC_ID          10   // Colour combination
#define SIPP_LUT_ID         11   // Look-up table
#define SIPP_DEHAZE_ID      19   // Dehaze
#define SIPP_TNF_ID         20   // Temporal Noise Filter

///////////////////////////////////////////////////////
// General purpose processing
#define SIPP_UPFIRDN0_ID    13   // Polyphase FIR filter[0]
#define SIPP_UPFIRDN1_ID    14   // Polyphase FIR filter[1]
#define SIPP_UPFIRDN2_ID    15   // Polyphase FIR filter[2]

///////////////////////////////////////////////////////
// Following are input buffer IDs

#define SIPP_LUMA_GI_ID     16   // Luma denoise - gain info
#define SIPP_CHR_LUM_EXP_ID 17   // Chroma denoise - Luma HDR long / middle exposure
#define SIPP_CC_CHROMA_ID   18   // Colour combination - chroma buffer
#define SIPP_SHARPEN_MM_ID  19   // Sharpen Motion Mask
#define SIPP_LSC_GM_ID      20   // Lens shading correction - gain mesh buffer
#define SIPP_RAW_DEFECT_ID  21   // RAW filter - defect pixel list for static defect correction - Base only
#define SIPP_LCA_TM_ID      22   // LCA filter - Transform mesh - Base only
#define SIPP_CC_3DLUT_ID    23   // Colour combination - 3D LUT buffer  - Base only
#define SIPP_LUT_LOAD_ID    24   // LUT filter - LUT buffer  - Base only
#define SIPP_SHARPEN_LUT_ID 25   // Sharpen Radial LUT - Base only
#define SIPP_DOG_SCD_ID     26   // The DoG secondary input

#define SIPP_LF_EXP0_ID     32   // Luma Fusion exposure 0
#define SIPP_LF_EXP1_ID     33   // Luma Fusion exposure 1
#define SIPP_LF_EXP2_ID     34   // Luma Fusion exposure 2
#define SIPP_TM_FUSE_LUM_ID 35   // Tone Map Fused Luma
#define SIPP_TM_GAIN_INF_ID 36   // Tone Map Fused Gain Info
#define SIPP_TM_SS_MEANS_ID 37   // Tone Map Sub-sampled means
#define SIPP_CF_ALPHA0_ID   38   // Chroma Fusion Alpha 0
#define SIPP_CF_EXP0_ID     39   // Chroma Fusion exposure 0
#define SIPP_CF_EXP1_ID     40   // Chroma Fusion exposure 1
#define SIPP_CF_EXP2_ID     41   // Chroma Fusion exposure 2
#define SIPP_DEHAZE_PRIN_ID 42   // De-Haze Primary
#define SIPP_DEHAZE_SSM_ID  43   // De-Haze Sub sampled means
#define SIPP_TNF_CUR_Y_ID   45   // TNF Current Y
#define SIPP_TNF_CUR_UV_ID  46   // TNF Current UV
#define SIPP_TNF_MM_ID      47   // TNF Motion Mask
#define SIPP_TNF_Y_PREV_ID  48   // TNF Previous Y
#define SIPP_TNF_UV_PREV_ID 49   // TNF Previous UV

///////////////////////////////////////////////////////
// Following are output buffer IDs
#define SIPP_DBYR_LUMA_ID   16   // Debayer luma buffer
#define SIPP_CHR_CF_ALP_ID  17   // Chroma Denoise HDR Chroma Fusion Alpha
#define SIPP_AE_STATS_ID    18   // RAW AE statistics - Base only
#define SIPP_AF_STATS_ID    19   // RAW AF statistics - Base only
#define SIPP_LUMA_HIST_ID   20   // RAW Luma histogram
#define SIPP_RGB_HIST_ID    21   // RAW RGB histogram
#define SIPP_PDAF_PIX_ID    22   // RAW PDAF Pixel
#define SIPP_FLICK_ACCUM_ID 23   // RAW Flicker row accumulation
#define SIPP_LF_FUSE_LUM_ID 32   // Luma Fusion Fused Luma
#define SIPP_LF_GAIN_INF_ID 33   // Luma Fusion Fused Gain Info
#define SIPP_LF_SS_MEANS_ID 34   // Luma Fusion Sub-sampled means
#define SIPP_TM_LUMA_ID     35   // Tone Map Luma
#define SIPP_TM_GI_ID       36   // Tone Map Gain Info
#define SIPP_CF_FCHROMA_ID  38   // Chroma Fusion Fused Chroma
#define SIPP_DHZ_PRIM_ID    42   // DeHaze Primary
#define SIPP_DHZ_SSM_ID     43   // DeHaze Sub Sampled Means
#define SIPP_DHZ_AL_STAT_ID 44   // DeHaze Air-Light Stats - Base only
#define SIPP_TNF_Y_ID       45   // TNF Y
#define SIPP_TNF_UV_ID      46   // TNF UV
#define SIPP_TNF_MM_ID      47   // Motion Mask info
#define SIPP_TNF_UV_2_ID    48   // Secondary UV (enables simultaneous write to DDR and to a CMX circ buffer in combo with SIPP_TNF_UV_ID)

///////////////////////////////////////////////////////
// General purpose processing

// Maximum IDs
#define SIPP_MAX_ID        SIPP_TNF_ID
#define SIPP_MIN_FILTER_ID SIPP_SIGMA_ID
#define SIPP_MAX_FILTER_ID SIPP_TNF_ID

#define SIPP_MAX_STREAM_ISP_OUT_ID SIPP_TNF_UV_2_ID

#define SIPP_HWISP_FILTER_MASK_SIZE ((SIPP_MAX_FILTER_ID - SIPP_MIN_FILTER_ID + 32)/32)

// Reserved IDs
#define SIPP_SIGMA_ID_MASK               (1 << SIPP_SIGMA_ID)
#define SIPP_RAW_ID_MASK                 (1 << SIPP_RAW_ID)
#define SIPP_STATS_MASK                  (1 << SIPP_STATS_ID)
#define SIPP_LCA_MASK                    (1 << SIPP_LCA_ID)
#define SIPP_LSC_ID_MASK                 (1 << SIPP_LSC_ID)
#define SIPP_LSC_GM_ID_MASK              (1 << SIPP_LSC_GM_ID)
#define SIPP_DBYR_ID_MASK                (1 << SIPP_DBYR_ID)
#define SIPP_CHROMA_ID_MASK              (1 << SIPP_CHROMA_ID)
#define SIPP_LUMA_ID_MASK                (1 << SIPP_LUMA_ID)
#define SIPP_SHARPEN_ID_MASK             (1 << SIPP_SHARPEN_ID)
#define SIPP_UPFIRDN0_ID_MASK            (1 << SIPP_UPFIRDN0_ID)
#define SIPP_UPFIRDN1_ID_MASK            (1 << SIPP_UPFIRDN1_ID)
#define SIPP_UPFIRDN2_ID_MASK            (1 << SIPP_UPFIRDN2_ID)
#define SIPP_MED_ID_MASK                 (1 << SIPP_MED_ID)
#define SIPP_LUT_ID_MASK                 (1 << SIPP_LUT_ID)
#define SIPP_LUT_LOAD_MASK               (1 << SIPP_LUT_LOAD_ID)
#define SIPP_CC_ID_MASK                  (1 << SIPP_CC_ID)
#define SIPP_CC_CHROMA_ID_MASK           (1 << SIPP_CC_CHROMA_ID)
#define SIPP_DOGD_ID_MASK                (1 << SIPP_DOG_ID)
#define SIPP_CGEN_ID_MASK                (1 << SIPP_CGEN_ID)

#define SIPP_LUMAFUS_ID_MASK             (1 << SIPP_LUMAFUS_ID)
#define SIPP_TONEMAP_ID_MASK             (1 << SIPP_TONEMAP_ID)
#define SIPP_CHRFUS_ID_MASK              (1 << SIPP_CHRFUS_ID)
#define SIPP_DEHAZE_ID_MASK              (1 << SIPP_DEHAZE_ID)
#define SIPP_TNF_ID_MASK                 (1 << SIPP_TNF_ID)

#define SIPP_DMA_ID                      (SIPP_MAX_ID+1  )
#define SIPP_MIPI_RX_ID                  (SIPP_MAX_ID+2  )
#define SIPP_MIPI_TX_ID                  (SIPP_MAX_ID+3  )
#define SIPP_SVU_ID                      (SIPP_MAX_ID+4  )
#define EXE_NUM                          (SIPP_MAX_ID+4+1)
#define SIPP_FAKE_ID                     (EXE_NUM + 1)

// Mask of HW filters which have LLBs and which should delay
// processing until the full kernel is in the LLB
// DoG, CGen and CComb are not in this list because they are
// not regularly scheduled and so may be handled some other way
#define SIPP_FILTER_LLB_MASK             (SIPP_SIGMA_ID_MASK | \
                                          SIPP_LSC_ID_MASK   | \
                                          SIPP_RAW_ID_MASK   | \
                                          SIPP_DBYR_ID_MASK  | \
                                          SIPP_LCA_MASK      | \
                                          SIPP_LUMA_ID_MASK  | \
                                          SIPP_DOGD_ID_MASK   | \
                                          SIPP_SHARPEN_ID_MASK | \
                                          SIPP_MED_ID_MASK   | \
                                          SIPP_CHROMA_ID_MASK | \
                                          SIPP_LUMAFUS_ID_MASK | \
                                          SIPP_TONEMAP_ID_MASK | \
                                          SIPP_CHRFUS_ID_MASK | \
                                          SIPP_DEHAZE_ID_MASK | \
                                          SIPP_TNF_ID_MASK)

#define SIPP_FILTER_FULL_LLB_MASK        (SIPP_SIGMA_ID_MASK | \
                                          SIPP_LSC_ID_MASK   | \
                                          SIPP_RAW_ID_MASK   | \
                                          SIPP_LCA_MASK      | \
                                          SIPP_DBYR_ID_MASK  | \
                                          SIPP_DOGD_ID_MASK   | \
                                          SIPP_LUMA_ID_MASK  | \
                                          SIPP_SHARPEN_ID_MASK | \
                                          SIPP_CGEN_ID_MASK  | \
                                          SIPP_MED_ID_MASK   | \
                                          SIPP_CHROMA_ID_MASK | \
                                          SIPP_LUMAFUS_ID_MASK | \
                                          SIPP_TONEMAP_ID_MASK | \
                                          SIPP_CHRFUS_ID_MASK | \
                                          SIPP_DEHAZE_ID_MASK | \
                                          SIPP_TNF_ID_MASK    | \
                                          SIPP_CC_ID_MASK)


///////////////////////////////////////////////////////
// CV Filter/buffer IDs
//
// TBD_KMB Do I need to offset this now?
#define CV_EDGE_OP_ID   32
#define CV_HARRIS_ID    33
#define CV_MEST_ID      34
#define CV_MIN_MAX_ID   35
#define CV_STEREO_ID    36
#define CV_WARP_0_ID    37
#define CV_WARP_1_ID    38
#define CV_UPFIRDN0_ID  39
#define CV_UPFIRDN1_ID  40
#define CV_UPFIRDN2_ID  41

// Following are input buffer IDs
#define CV_STEREO_IP1_BUF_ID  40
#define CV_MINMAX_IP1_BUF_ID  41
#define CV_MINMAX_IP2_BUF_ID  42
#define CV_WARP_0_IP1_BUF_ID  43
#define CV_WARP_0_IP2_BUF_ID  44
#define CV_WARP_1_IP1_BUF_ID  45
#define CV_WARP_1_IP2_BUF_ID  46

// Following are output buffer IDs
#define CV_STEREO_OP1_BUF_ID  40
#define CV_MINMAX_OP1_BUF_ID  41
#define CV_WARP_0_OP1_BUF_ID  42
#define CV_WARP_0_OP2_BUF_ID  43
#define CV_WARP_1_OP1_BUF_ID  44
#define CV_WARP_1_OP2_BUF_ID  45

#define CV_EDGE_ID_MASK      (0x1 << (CV_EDGE_OP_ID - 0x20))
#define CV_HARRIS_ID_MASK    (0x1 << (CV_HARRIS_ID - 0x20))
#define CV_MEST_ID_MASK      (0x1 << (CV_MEST_ID - 0x20))
#define CV_MIN_MAX_ID_MASK   (0x1 << (CV_MIN_MAX_ID - 0x20))
#define CV_STEREO_ID_MASK    (0x1 << (CV_STEREO_ID - 0x20))
#define CV_WARP_0_ID_MASK    (0x1 << (CV_WARP_0_ID - 0x20))
#define CV_WARP_1_ID_MASK    (0x1 << (CV_WARP_1_ID - 0x20))
#define CV_UPFIRDN0_ID_MASK  (0x1 << (CV_UPFIRDN0_ID - 0x20))
#define CV_UPFIRDN1_ID_MASK  (0x1 << (CV_UPFIRDN1_ID - 0x20))
#define CV_UPFIRDN2_ID_MASK  (0x1 << (CV_UPFIRDN2_ID - 0x20))

#define CV_STEREO_IP1_ID_MASK  (0x1 << (CV_STEREO_IP1_BUF_ID - 0x20))
#define CV_MINMAX_IP1_ID_MASK  (0x1 << (CV_MINMAX_IP1_BUF_ID - 0x20))
#define CV_MINMAX_IP2_ID_MASK  (0x1 << (CV_MINMAX_IP2_BUF_ID - 0x20))
#define CV_WARP_0_IP1_ID_MASK  (0x1 << (CV_WARP_0_IP1_BUF_ID - 0x20))
#define CV_WARP_0_IP2_ID_MASK  (0x1 << (CV_WARP_0_IP2_BUF_ID - 0x20))
#define CV_WARP_1_IP1_ID_MASK  (0x1 << (CV_WARP_1_IP1_BUF_ID - 0x20))
#define CV_WARP_1_IP2_ID_MASK  (0x1 << (CV_WARP_1_IP2_BUF_ID - 0x20))

#define CV_STEREO_OP1_ID_MASK  (0x1 << (CV_STEREO_OP1_BUF_ID - 0x20))
#define CV_MINMAX_OP1_ID_MASK  (0x1 << (CV_MINMAX_OP1_BUF_ID - 0x20))
#define CV_WARP_0_OP1_ID_MASK  (0x1 << (CV_WARP_0_OP1_BUF_ID - 0x20))
#define CV_WARP_0_OP2_ID_MASK  (0x1 << (CV_WARP_0_OP2_BUF_ID - 0x20))
#define CV_WARP_1_OP1_ID_MASK  (0x1 << (CV_WARP_1_OP1_BUF_ID - 0x20))
#define CV_WARP_1_OP2_ID_MASK  (0x1 << (CV_WARP_1_OP2_BUF_ID - 0x20))

#define CV_EDGE_INPUT_MASK    CV_EDGE_ID_MASK
#define CV_HARRIS_INPUT_MASK  CV_HARRIS_ID_MASK
#define CV_MEST_INPUT_MASK    CV_MEST_ID
#define CV_MINMAX_INPUT_MASK  (CV_MIN_MAX_ID_MASK | CV_MINMAX_IP1_ID_MASK | CV_MINMAX_IP2_ID_MASK)
#define CV_STEREO_INPUT_MASK  (CV_STEREO_ID_MASK | CV_STEREO_IP1_ID_MASK)
#define CV_WARP_0_INPUT_MASK  (CV_WARP_0_ID_MASK | CV_WARP_0_IP1_ID_MASK | CV_WARP_0_IP2_ID_MASK)
#define CV_WARP_1_INPUT_MASK  (CV_WARP_1_ID_MASK | CV_WARP_1_IP1_ID_MASK | CV_WARP_1_IP2_ID_MASK)

#define CV_EDGE_OUTPUT_MASK    CV_EDGE_ID_MASK
#define CV_HARRIS_OUTPUT_MASK  CV_HARRIS_ID_MASK
#define CV_MEST_OUTPUT_MASK    CV_MEST_ID
#define CV_MINMAX_OUTPUT_MASK  (CV_MIN_MAX_ID_MASK | CV_MINMAX_OP1_ID_MASK)
#define CV_STEREO_OUTPUT_MASK  (CV_STEREO_ID_MASK | CV_STEREO_OP1_ID_MASK)
#define CV_WARP_0_OUTPUT_MASK  (CV_WARP_0_ID_MASK | CV_WARP_0_OP1_ID_MASK | CV_WARP_0_OP2_ID_MASK)
#define CV_WARP_1_OUTPUT_MASK  (CV_WARP_1_ID_MASK | CV_WARP_1_OP1_ID_MASK | CV_WARP_1_OP2_ID_MASK)

#define SIPP_FILTER_CV_LLB_MASK  (CV_HARRIS_ID_MASK | \
                                  CV_MIN_MAX_ID_MASK | \
                                  CV_STEREO_ID_MASK)

///////////////////////////////////////////////////////
// General purpose processing

#define SIPP_MIN_CV_ID   CV_EDGE_OP_ID
#define SIPP_MAX_CV_ID   CV_UPFIRDN2_ID

#define SIPP_MIN_WARP_ID CV_WARP_0_ID
#define SIPP_MAX_WARP_ID CV_WARP_1_ID
#define SIPP_MAX_HW_ID   SIPP_MAX_CV_ID

#define SIPP_HWCV_FILTER_MASK_SIZE ((SIPP_MAX_CV_ID - SIPP_MIN_CV_ID + 32)/32)
#define SIPP_HWCV_FILTER_NUM       (SIPP_MAX_CV_ID - SIPP_MIN_CV_ID + 1)

#define SIPP_HWCV_NUM_WARP_UNITS   (SIPP_MAX_WARP_ID - SIPP_MIN_WARP_ID + 1)

#define SIPP_NUM_CV_UNITS (SIPP_MAX_CV_ID - SIPP_MIN_CV_ID + 1)

#define SIPP_NUM_MANAGED_IRQS  (SIPP_NUM_CV_UNITS)

#define SIPP_MGD_IRQ_EDGE    (CV_EDGE_OP_ID - 0x20)
#define SIPP_MGD_IRQ_HARRIS  (CV_HARRIS_ID - 0x20)
#define SIPP_MGD_IRQ_MEST    (CV_MEST_ID - 0x20)
#define SIPP_MGD_IRQ_MIN_MAX (CV_MIN_MAX_ID - 0x20)
#define SIPP_MGD_IRQ_STEREO  (CV_STEREO_ID - 0x20)
#define SIPP_MGD_IRQ_WARP_0  (CV_WARP_0_ID - 0x20)
#define SIPP_MGD_IRQ_WARP_1  (CV_WARP_1_ID - 0x20)

///////////////////////////////////////////////////////
// Virtual Filters
// Add concepts for SIPP sink wrappers of JPEG



////////////////////////////////////////////////////
// OSE Macros

#define SIPP_FREE_RT_FILTER_LIST ((1 << SIPP_SIGMA_ID)    |\
                                  (1 << SIPP_LSC_ID)      |\
                                  (1 << SIPP_RAW_ID)      |\
                                  (1 << SIPP_LCA_ID)      |\
                                  (1 << SIPP_DBYR_ID)     |\
                                  (1 << SIPP_LUMAFUS_ID)  |\
                                  (1 << SIPP_TONEMAP_ID)  |\
                                  (1 << SIPP_LUMA_ID)     |\
                                  (1 << SIPP_DOG_ID)      |\
                                  (1 << SIPP_CGEN_ID)     |\
                                  (1 << SIPP_MED_ID)      |\
                                  (1 << SIPP_CHROMA_ID)   |\
                                  (1 << SIPP_CHRFUS_ID)   |\
                                  (1 << SIPP_CC_ID)       |\
                                  (1 << SIPP_DEHAZE_ID)   |\
                                  (1 << SIPP_LUT_ID)      |\
                                  (1 << SIPP_TNF_ID)      |\
                                  (1 << SIPP_SHARPEN_ID)  |\
                                  (1 << SIPP_UPFIRDN0_ID) |\
                                  (1 << SIPP_UPFIRDN1_ID) |\
                                  (1 << SIPP_UPFIRDN2_ID))

#define SIPP_DMA_CLIENT_ID      0xD17AD17A

#endif // _SIPP_HW_IDS_2X9X_H_

