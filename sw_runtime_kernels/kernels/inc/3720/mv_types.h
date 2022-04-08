//
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
///
/// @file
///
/// @brief     Basic type definitions
///

#include <stdbool.h>
#include <stdint.h>

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

#if defined(__PC__) && !defined(X86_BUILD_FOR_HOST_BLOB_PARSING)
#include <half.h>
#endif

#if defined(__leon_rt__) || defined(__leon_nn__) || defined(__arm__) || defined(__aarch64__) || \
        defined(X86_BUILD_FOR_HOST_BLOB_PARSING)
typedef int16_t half;
#elif defined(__shave__) /* shave_nn doesn't support vector types */
#include <moviVectorTypes.h>
#include <stdbool.h>
#endif

typedef half fp16;

typedef float fp32;

typedef struct {
    uint64_t cmxRamLayoutCfg0;
    uint64_t cmxRamLayoutCfg1;
} CmxRamLayoutCfgType;

typedef union {
    uint32_t u32;
    fp32 f32;
} u32f32;

// 3: Local const declarations     NB: ONLY const declarations go here
// ----------------------------------------------------------------------------

#endif /* _MV_TYPES_H_ */
