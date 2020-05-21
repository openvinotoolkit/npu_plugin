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
/// @file      swcMutexUsage.h
/// 
/// @copyright All code copyright Movidius Ltd 2013, all rights reserved
///            For License Warranty see: common/license.txt
///
/// @brief     Contains a list of all Mutexes used in the MDK. Currently manually generated. Many of these are
/// @brief     configurable by the application at build time. See additional comments for information
/// @brief     TODO: have automatic generation of this. Meanwhile: please submit movidius.org ticket
/// @brief     TODO: if you find any used mutex missing from here. All marked "UNUSED" may be used at user level.
///

#ifndef COMMON_SHARED_INCLUDE_SWCMUTEXUSAGE_H_
#define COMMON_SHARED_INCLUDE_SWCMUTEXUSAGE_H_

//Mutex 0 usage not configurable. Used by SIPP
#define SIPP_MUTEX_0             0
//Mutex 0 usage for VCS printf. VCS mutex'd printf/SIPP usage is exclusive only
#define VCS_MUTEX_PRINTF0        0
//Mutex 1 usage not configurable. Used by SIPP
#define SIPP_MUTEX_1             1
#define UNUSED_MUTEX_2           2
#define UNUSED_MUTEX_3           3
#define UNUSED_MUTEX_4           4
#define UNUSED_MUTEX_5           5
#define UNUSED_MUTEX_6           6
#define UNUSED_MUTEX_7           7
#define UNUSED_MUTEX_8           8
#define UNUSED_MUTEX_9           9
#define UNUSED_MUTEX_10         10
#define UNUSED_MUTEX_11         11
#define UNUSED_MUTEX_12         12
#define MEM_MGR_MUTEX           13
#define DTB_CV_API_MUTEX        14
#define WARP1_CV_API_MUTEX      15
#define WARP2_CV_API_MUTEX      16
#define WARP3_CV_API_MUTEX      17
#define STEREO_CV_API_MUTEX     18
#define MOTEST_CV_API_MUTEX     19
#define EDGE_CV_API_MUTEX       20
#define HARRIS_CV_API_MUTEX     21
#define MIN_MAX_CV_API_MUTEX    22
//Mutex 23 used by the Dynamic Loading Process. Not configurable.
#define DYNCONTEXT_MUTEX_23     23
//Mutex 24 used by the CMXDMA driver. Configurable at build time by defining DRV_CMX_DMA_MUTEX_ID_LA_0
#define CMXDMA_M0_MUTEX_24      24
//Mutex 25 used by the CMXDMA driver. Configurable at build time by defining DRV_CMX_DMA_MUTEX_ID_LA_1
#define CMXDMA_M1_MUTEX_25      25
//Mutex 26 used by the CMXDMA driver. Configurable at build time by defining DRV_CMX_DMA_MUTEX_ID_LA_2
#define CMXDMA_M2_MUTEX_26      26
//Mutex 27 used by the CMXDMA driver. Configurable at build time by defining DRV_CMX_DMA_MUTEX_ID_LA_3
#define CMXDMA_M3_MUTEX_27      27
//Mutex 28 used by the L2 cache driver. Configurable at build time by defining SHAVE_L2C_HW_MUTEX_USED
#define SHAVE_L2C_MUTEX_28      28
//Mutex 29 used by ResMgr for resource lock. Not configurable.
#define RESMGR_RES_MUTEX_29    29
//Mutex 30 used by ResMgr for level 0 lock. Not configurable
#define RESMGR_L0_MUTEX_30     30
//Mutex 31 used by ResMgr for level 1 lock. Not configurable
#define RESMGR_L1_MUTEX_31     31


#endif /* COMMON_SHARED_INCLUDE_SWCMUTEXUSAGE_H_ */
