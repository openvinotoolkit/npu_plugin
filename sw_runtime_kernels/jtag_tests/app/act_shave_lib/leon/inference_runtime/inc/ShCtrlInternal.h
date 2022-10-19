//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef SHCTRLINTERNAL_H
#define SHCTRLINTERNAL_H

#include <HglShaveCommon.h>

#if defined(SHAVE_PROCESSOR_MAIN)

#include <stdint.h>
#include <stdbool.h>
#include <rtems.h>

#define ShaveType HglShaveType

typedef struct ShHandle {
    // Persistent members
    uint8_t *base; // base address of shave: _must_ be first element
    ShaveType type;
    uint32_t id;
    rtems_id waitSema; // sema to be used in wait functions

    // Non-persistent members
    uint32_t reason;        // reason for completion of payload
    uintptr_t stackAddr;    // Stack pointer address (8 byte aligned down)
    uint32_t stackSize;     // Stack size limit in bytes (8 byte aligned down)
    uintptr_t winOffset[4]; // Shave Window Offsets (1024 byte aligned)
    bool opened;            // true if we have openend the handle
    bool usingPoll;         // true if ShaveCtrlWait polls for completion
    bool collected;         // true if we did the post-run collect stage

    union {
        union {
            uint32_t ret_uint32_t;
            uint64_t ret_uint64_t;
            void *ret_voidPtr;
            uint8_t ret_bytes[16];
            uint32_t ret_uints[4];
        };
        struct {
            uint32_t irfReturnData[4];
        };
    };
} ShHandle;

#define SHAVE_PARAMS_FIRST_IRF 18
#define SHAVE_PARAMS_LAST_IRF 11
#define SHAVE_PARAMS_IRF_COUNT (SHAVE_PARAMS_FIRST_IRF - SHAVE_PARAMS_LAST_IRF + 1)
#define SHAVE_PARAMS_FIRST_VRF 23
#define SHAVE_PARAMS_LAST_VRF 16
#define SHAVE_PARAMS_VRF_COUNT (SHAVE_PARAMS_FIRST_VRF - SHAVE_PARAMS_LAST_VRF + 1)
#define SHAVE_PARAMS_COUNT_LIMIT 64

#define SHAVE_RETURN_FIRST_IRF 18
#define SHAVE_RETURN_DOWNWARDS_IRF

#define SH_ADDR_WINDOW_ALIGN_MASK (0x3FF)

#endif

#endif /* SHCTRLINTERNAL_H */
/// @}
