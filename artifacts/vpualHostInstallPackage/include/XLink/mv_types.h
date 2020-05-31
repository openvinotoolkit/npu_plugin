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
/// @file      mv_types.h
/// 
/// @copyright All code copyright Movidius Ltd 2012, all rights reserved
///            For License Warranty see: common/license.txt
///
/// @brief     Basic type definitions
///

#ifndef _MV_TYPES_H_
#define _MV_TYPES_H_

// 1: Defines
// ----------------------------------------------------------------------------

#ifndef FALSE
#define FALSE (0)
#endif

#ifndef TRUE
#define TRUE (1)
#endif

#ifndef NULL
#define NULL (0)
#endif

#define ALL_ZEROS (0x00000000)
#define ALL_ONES (0xFFFFFFFF)

/* Limits of integral types.  */


// 2: Typedefs (types, enums, structs)
// ----------------------------------------------------------------------------

// mv typedefs
#include <stdint.h>

#ifdef __shave__
#include <moviVectorTypes.h>
#endif

// Define for unused variables
#define UNUSED(x) (void)x

typedef uint8_t u8;
typedef int8_t s8;
typedef uint16_t u16;
typedef int16_t s16;
typedef uint32_t u32;
typedef int32_t s32;
typedef uint64_t u64;
typedef int64_t s64;

typedef int16_t half;
typedef half fp16;

typedef float fp32;

typedef struct
{
    u64 cmxRamLayoutCfg0;
    u64 cmxRamLayoutCfg1;
} CmxRamLayoutCfgType;

typedef union
{
    uint32_t u32;
    float f32;
}u32f32;

// 3: Local const declarations     NB: ONLY const declarations go here
// ----------------------------------------------------------------------------

#endif /* _MV_TYPES_H_ */
