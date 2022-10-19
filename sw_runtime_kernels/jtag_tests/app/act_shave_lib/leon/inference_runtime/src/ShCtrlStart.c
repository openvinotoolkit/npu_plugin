//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#include <ShCtrl.h>
#include <stdarg.h>
#include <HglShaveCommon.h>
#include <mvLog.h>

#define SH_PARAMS_START_IRF 18
#define SH_PARAMS_NIRFS 8
#define SH_PARAMS_END_IRF (SH_PARAMS_START_IRF - SH_PARAMS_NIRFS + 1)
#define SH_ADDR_WINDOW_NBITS (24)
#define SH_ADDR_WINDOW_SIZE (1 << SH_ADDR_WINDOW_NBITS)
#define SH_ADDR_WINDOW_SIZE_MASK (SH_ADDR_WINDOW_SIZE - 1)
#define SHAVE_ADDR_WINDOW_A (0x1C000000U)

static uint32_t *getStackPointer(ShHandle *handle) {
    uintptr_t winIdx;
    uintptr_t addr = handle->stackAddr - 8;

    /// The stack address is deducted by 8 to adjust it to its first "push"
    /// position. This avoids edge conditions where SP = top of window + 1

    winIdx = addr >> SH_ADDR_WINDOW_NBITS;
    winIdx = winIdx - (SHAVE_ADDR_WINDOW_A >> SH_ADDR_WINDOW_NBITS);
    if (winIdx >= HGL_SHAVE_WINDOW_NB)
        return (uint32_t *)handle->stackAddr;

    // Inside a window
    if (handle->winOffset[winIdx] == 0)
        return NULL;

    // Convert addr relative to window, add the window offset, add back the 8
    addr = (addr & SH_ADDR_WINDOW_SIZE_MASK) + handle->winOffset[winIdx] + 8;
    return (uint32_t *)addr;
}

static HglShaveCtrlError getRequiredStackBytes(const char *fmt, bool hasIrf, uint32_t *result) {
    uint32_t stackBytes = 0;

    for (; *fmt; fmt++) {
        switch (*fmt) {
            case ' ':
                break;
            case 'p':
            case 'i':
            case 'u':
                if (hasIrf)
                    hasIrf = false;
                else
                    stackBytes += 4;
                break;
            case 'w':
                stackBytes = (stackBytes + 7) & ~7;
                stackBytes += 8;
                break;
            default:
                mvLog(MVLOG_ERROR, "Unknown specifier: '%c' in format string", *fmt);
                return HGL_SHAVE_CTRL_PARAMETER_ERROR;
        }
    }
    *result = (stackBytes + 7) & ~7;
    return HGL_SHAVE_CTRL_SUCCESS;
}

static HglShaveCtrlError parseParametersToStack(ShHandle *handle, const char *fmt, va_list vargs, bool hasIrf) {
    uint32_t *stackPtr;
    uint32_t stackBytes;

    if (!handle->stackAddr) {
        mvLog(MVLOG_ERROR, "No stack is available for Shave parameters");
        return HGL_SHAVE_CTRL_NO_STACK;
    }

    if (getRequiredStackBytes(fmt, hasIrf, &stackBytes) != HGL_SHAVE_CTRL_SUCCESS)
        return HGL_SHAVE_CTRL_PARAMETER_ERROR;

    if (handle->stackSize && handle->stackSize < stackBytes) {
        mvLog(MVLOG_ERROR, "Stack is not large enough for Shave parameters");
        return HGL_SHAVE_CTRL_STACK_TOO_SMALL;
    }

    // Get the current CPU side Shave stack pointer (the stackAddr address may
    // be translated through a window). Adjust this pointer down by the number
    // of uint32_ts needed for the parameters (NOT the number of bytes)

    stackPtr = getStackPointer(handle);
    if (!stackPtr) {
        mvLog(MVLOG_ERROR, "Stack pointer translates through an unset Shave Window");
        return HGL_SHAVE_CTRL_NO_STACK;
    }
    stackPtr -= stackBytes / sizeof(uint32_t);

    // Store the adjusted stack address and optional size into Shave registers:
    uintptr_t newStackAddr = handle->stackAddr - stackBytes;
    uintptr_t newStackSize = 0;
    if (handle->stackSize)
        newStackSize = handle->stackSize - stackBytes;

    HglShaveSetStack(handle->base, (uint32_t)newStackAddr, (uint32_t)newStackSize);

    // Push parameters on to the stack; 64 bit params must be 8 byte aligned
    bool aligned = true;
    uint32_t val32;
    uint64_t val64;

    for (; *fmt; fmt++) {
        switch (*fmt) {
            case 'p':
                mvLog(MVLOG_WARN, "Format specifier 'p' is deprecated");
                // Intentionally falls through
            case 'i':
            case 'u':
                val32 = va_arg(vargs, uint32_t);
                if (hasIrf) {
                    hasIrf = false;
                    HglShaveSetIrf(handle->base, SH_PARAMS_END_IRF, val32);
                } else {
                    *stackPtr++ = val32;
                    aligned = !aligned;
                }
                break;

            case 'w':
                // low word in lowest memory address; then high word
                val64 = va_arg(vargs, uint64_t);
                if (!aligned) {
                    *stackPtr++ = 0;
                    aligned = true;
                }
                *stackPtr++ = (uint32_t)val64;
                *stackPtr++ = (uint32_t)(val64 >> 32);
                break;

            default:
                break;
        }
    }

    return HGL_SHAVE_CTRL_SUCCESS;
}

static HglShaveCtrlError parseParameters(ShHandle *handle, const char *fmt, va_list vargs) {
    uint32_t val32;
    uint64_t val64;
    int nirfs = SH_PARAMS_NIRFS;
    uint32_t irfIndex = SH_PARAMS_START_IRF;

    for (; *fmt; fmt++) {
        switch (*fmt) {
            case ' ':
                break;

            case 'p': // 32 bit pointer (deprecated; VPU4 pointers are 64 bit)
                mvLog(MVLOG_WARN, "Format specifier 'p' is deprecated");
                // Intentionally falls through
            case 'i': // 32 bit integer
            case 'u': // 32 bit unsigned integer
                if (nirfs < 1)
                    return parseParametersToStack(handle, fmt, vargs, false);
                nirfs--;
                val32 = va_arg(vargs, uint32_t);
                HglShaveSetIrf(handle->base, irfIndex--, val32);
                break;

            case 'w':
                // wide = 64 bits
                // the high numbered register contains the high 32-bits
                // the low numbered register contains the low 32-bits
                if (nirfs < 2)
                    return parseParametersToStack(handle, fmt, vargs, true);
                nirfs -= 2;
                val64 = va_arg(vargs, uint64_t);
                HglShaveSetIrf(handle->base, irfIndex--, (uint32_t)(val64 >> 32));
                HglShaveSetIrf(handle->base, irfIndex--, (uint32_t)val64);
                break;

            default:
                mvLog(MVLOG_ERROR, "Unknown specifier: '%c' in format string", *fmt);
                return HGL_SHAVE_CTRL_PARAMETER_ERROR;
        }
    }

    return HGL_SHAVE_CTRL_SUCCESS;
}

HglShaveCtrlError ShCtrlStart(ShHandle *handle, void *startAddr, const char *fmt, ...) {
    va_list vargs;
    HglShaveCtrlError err;

    if (!handle || !startAddr || !fmt) {
        mvLog(MVLOG_ERROR, "Bad parameter");
        return HGL_SHAVE_CTRL_PARAMETER_ERROR;
    }
    if (!handle->stackAddr)
        mvLog(MVLOG_WARN, "Shave has no stack");

#ifdef SHAVE_WAIT_POLL
    handle->usingPoll = true;
#endif

    // Set the stack address, stack size and the Shave Windows from their
    // cached values in the handle. They are all aligned correctly
    HglShaveSetStack(handle->base, (uint32_t)handle->stackAddr, (uint32_t)handle->stackSize);
    HglShaveSetWindowOffsets(handle->base, handle->winOffset);

    va_start(vargs, fmt);
    err = parseParameters(handle, fmt, vargs);
    va_end(vargs);
    if (err != HGL_SHAVE_CTRL_SUCCESS)
        return err;

    handle->collected = false; // Collect return statuses in ShaveCtrlWait
    HglShaveSetAndStart(handle->base, (uintptr_t)startAddr, handle->usingPoll);
    return HGL_SHAVE_CTRL_SUCCESS;
}
