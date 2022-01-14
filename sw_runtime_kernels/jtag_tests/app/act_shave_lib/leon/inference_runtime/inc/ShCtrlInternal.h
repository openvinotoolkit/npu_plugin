//
// Copyright Intel Corporation.
//
// LEGAL NOTICE: Your use of this software and any required dependent software
// (the "Software Package") is subject to the terms and conditions of
// the Intel(R) OpenVINO(TM) Distribution License for the Software Package,
// which may also include notices, disclaimers, or license terms for
// third party or open source software included in or with the Software Package,
// and your use indicates your acceptance of all such terms. Please refer
// to the "third-party-programs.txt" or other similarly-named text file
// included with the Software Package for additional details.
//

#ifndef SHCTRLINTERNAL_H
#define SHCTRLINTERNAL_H

#include <HglShaveCommon.h>

#if defined(SHAVE_PROCESSOR_MAIN)

#include <stdint.h>
#include <stdbool.h>
#include <rtems.h>

typedef struct ShHandle {
    HglShaveType type;
    uint32_t id;
    uint8_t *base;         // base address of shave
    rtems_id waitSema;     // sema to be used in wait functions
    uint8_t resetBegin[0]; // label from where to zero in case of reset
    // From here, all members are subjected to reset
    uint8_t *bss;          // start of bss
    uint32_t bssSize;      // size in bytes of bss
    uint32_t reason;       // reason for completion of payload
    uint32_t *paramSP;     // current stack pointer
    uint32_t availableIrf; // current available IRF count for parameters
    uint32_t availableVrf; // current available VRF count for parameters
    bool opened;           // true if we have openend the handle
    bool hasStack;         // true if we have defined stack
    bool hasStackSize;     // true if we have defined stack size
    bool collected;        // true if we did the post-run collect stage

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
    uint8_t resetEnd[0]; // label to end of reset to zero zone
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

#endif

#endif /* SHCTRLINTERNAL_H */
/// @}
