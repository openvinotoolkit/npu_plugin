// 
// ///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
/// @file 		sippBaseTypes.h
/// @copyright 	All code copyright Movidius Ltd 2015, all rights reserved.
///            	For License Warranty see: common/license.txt
///
/// @brief     	Public header file for SIPP base types
//             	Aim to replace with a generic moviTypes.h file if one exists
//
//

#ifndef _SIPP_BASE_TYPES_H_
#define _SIPP_BASE_TYPES_H_

//////////////////////////////////////////////////////////////
// SIPP Base Types

#ifdef SIPP_USE_OWN_BASETYPES
// till MDK people wake up, I put these defs here...
// they should be in mvtypes
typedef unsigned long long UInt64;
typedef unsigned int       UInt32;
typedef unsigned short     UInt16;
typedef unsigned char      UInt8;

typedef          int       Int32;
typedef          short     Int16;
typedef          char      Int8;

#ifndef __cplusplus
typedef          uint8_t        bool;
#endif
#else

#ifdef __myriad2__
#include <mv_types.h>
#if SIPP_RTOS == SIPP_NO_RTOS
#ifndef __cplusplus
typedef          uint8_t        bool;
#endif
#endif
#else
// For the c-model, some HW model header files are incompatible with mv_types.h
#include <stdint.h>
typedef  uint8_t                                      uint8_t;
typedef   int8_t                                      int8_t;
typedef uint16_t                                      uint16_t;
typedef  int16_t                                      int16_t;
typedef uint32_t                                      uint32_t;
typedef  int32_t                                      int32_t;
typedef uint64_t __attribute__((aligned(8)))          uint64_t;
typedef  int64_t __attribute__((aligned(8)))          s64;

typedef    float          fp32;

#ifndef __cplusplus
typedef uint8_t                 bool;
#endif
#endif
// Still define the 'old' basetypes as they have been exported to many existing
// applications and filter wrappers
typedef uint64_t           UInt64;
typedef uint32_t           UInt32;
typedef uint16_t           UInt16;
typedef uint8_t            UInt8;

typedef int32_t            Int32;
typedef int16_t            Int16;
typedef int8_t             Int8;

#endif



////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////
#if SIPP_RTOS == SIPP_NO_RTOS
#ifndef true
#define true 1
#endif
#ifndef false
#define false 0
#endif
#endif

#endif /* _SIPP_BASE_TYPES_H_ */

