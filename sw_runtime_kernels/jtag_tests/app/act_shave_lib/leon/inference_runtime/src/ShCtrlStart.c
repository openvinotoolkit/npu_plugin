// {% copyright %}
///
/// @file ShaveCtrlStart.c
/// @defgroup SvuCtrl
/// @{
/// @brief Shave Control System level driver
///

#include <ShCtrl.h>
#include <stdarg.h>

#define ALIGN_POINTER(XX, YY) ((uint32_t *)(((intptr_t)(XX) + YY - 1) & (~(YY - 1))))

static void shCtrlParamAddU32(ShHandle *handle, uint32_t value32) {
    SHAVE_FUNC("%p, %" PRId32"", handle, value32);
    // Parameters and conditions for failing are already checked in ShaveCtrlStart
    // This function cannot fail.

    if (handle->availableIrf > 0) { // do we have at least one free IRF register?
        handle->availableIrf--;     // use one IRF
        // LAST irf is actually the lowest one since it grows in decrements
        uint32_t irfIndex = handle->availableIrf + SHAVE_PARAMS_LAST_IRF;
        // and store it in one IRF
        HglShaveSetIrf((HglShaveHandle *)handle, irfIndex, value32);
    } else {
        // no available IRF so using stack
        *handle->paramSP++ = value32;
    }
}

static void shCtrlParamAddU64(ShHandle *handle, uint64_t value64) {
    SHAVE_FUNC("%p, %" PRId64"", handle, value64);
    // Parameters and conditions for failing are already checked in ShaveCtrlStart
    // This function cannot fail.
    uint32_t value32_1 = value64 >> 32;
    uint32_t value32_2 = value64 & 0xFFFFFFFF;

    if (handle->availableIrf > 1) { // do we have two free IRF registers?
        handle->availableIrf -= 2;  // use two IRF
        // LAST irf is actually the lowest one since it grows in decrements
        uint32_t irfIndex = handle->availableIrf + SHAVE_PARAMS_LAST_IRF;
        // and store it in IRF
        // we already made sure we have two available
        HglShaveSetIrf((HglShaveHandle *)handle, irfIndex + 0, value32_2);
        HglShaveSetIrf((HglShaveHandle *)handle, irfIndex + 1, value32_1);
    } else {
        // two IRF not available so using stack
        // align SP to 8 bytes
        handle->paramSP = ALIGN_POINTER(handle->paramSP, 8);
        *handle->paramSP++ = value32_2;
        *handle->paramSP++ = value32_1;
    }
}

static void shCtrlParamAddV64(ShHandle *handle, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    SHAVE_FUNC("%p, %" PRId32", %" PRId32", %" PRId32", %" PRId32"", handle, v0, v1, v2, v3);
    // do we have one free VRF register?
    if (handle->availableVrf > SHAVE_PARAMS_LAST_VRF) {
        handle->availableVrf -= 1; // use one VRF
        uint32_t vrfIndex = handle->availableVrf + SHAVE_PARAMS_LAST_VRF;
        HglShaveSetVrf((HglShaveHandle *)handle, vrfIndex, v0, v1, v2, v3);
    } else {
        // VRF not available so trying our luck with the stack
        // But we need to align the stack to 8 bytes
        handle->paramSP = ALIGN_POINTER(handle->paramSP, 8);
        *handle->paramSP++ = v0;
        *handle->paramSP++ = v1;
        *handle->paramSP++ = v2;
        *handle->paramSP++ = v3;
    }
}

// This function will count the bytes needed for the shave stack to hold the parameters
// It will also take into account that there are the IRF and VRF registers so no stack
// will be used for that. It does this by iterating over the parameter list in lst argument.
static HglShaveCtrlError shGetNeededStackForParams(const char *lst, uint32_t *stackSize) {
    SHAVE_FUNC("%p, %p", lst, stackSize);
    *stackSize = 0;
    uint32_t availableIRF = SHAVE_PARAMS_IRF_COUNT;
    uint32_t availableVRF = SHAVE_PARAMS_VRF_COUNT;
    for (;; lst++) {
        switch (*lst) {
            case '\0':
                SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SUCCESS);
            case ' ':
                continue;
            case 'i':
            case 'u':
            case 'p':
                if (availableIRF > 0) {
                    availableIRF--; // we either use one IRF register
                } else {
                    *stackSize += 4; // or use 4 bytes of stack
                }
                break;
            case 'w':
                if (availableIRF > 1) {
                    availableIRF -= 2; // we either use two IRF registers
                } else {
                    *stackSize += 8; // or use 8 bytes of stack
                }
                break;
            case 'v':
                // we support only 4x32 bits vectors
                if (availableVRF > 0) {
                    availableVRF--; // we either use a VRF register
                } else {
                    *stackSize += 16; // or use 16 bytes of stack
                }
                break;
            default:
                SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_PARAMETER_ERROR);
                break;
        }
    }
    SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SUCCESS);
}

static HglShaveCtrlError shPrepareAndStart(ShHandle *handle, void *entry) {
    SHAVE_FUNC("%p, %p", handle, entry);
    rtems_status_code sc = rtems_semaphore_obtain(handle->waitSema, RTEMS_NO_WAIT, 0);
    if (RTEMS_SUCCESSFUL != sc) {
        SHAVE_LOG(" - Failed due to rtems_semaphore_obtain with return code: %" PRId32"", sc);
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SEMA_FAILURE);
    }
    handle->collected = false; // trigger a new return collect when done

#ifndef SHAVE_WAIT_POLL
    ShCtrlIsrPrepare(handle);
#endif

    HglShaveSetAndStart((HglShaveHandle *)handle, entry);
    SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SUCCESS);
}

static HglShaveCtrlError shProcessParamsAndStart(ShHandle *handle, void *entry, const char *fmt, va_list list) {
    SHAVE_FUNC("%p, %p, %s, ...", handle, entry, fmt);
    const char *type = fmt;
    for (;; type++) {
        switch (*type) {
            case '\0':
                SHAVE_RETURN_ERR(shPrepareAndStart(handle, entry));
            case ' ':
                continue;
            case 'i':
                // integer (same as 32 bits)
            case 'u':
                // unsigned (same as 32 bits)
            case 'p':;
                // pointer (same as 32 bits)
                uint32_t value32 = va_arg(list, uint32_t);
                shCtrlParamAddU32(handle, value32);
                break;
            case 'w':;
                // "wide" integer aka 64 bits
                uint64_t value64 = va_arg(list, uint64_t);
                shCtrlParamAddU64(handle, value64);
                break;
            case 'v':;
                // for now it is the responsability of the user to check if the shave requested
                // actually has VRF registers. In the future they the driver will automatically
                // populate IRF instead of VRF, if movicompile supports it (it should)
                uint32_t v0 = va_arg(list, uint32_t);
                uint32_t v1 = va_arg(list, uint32_t);
                uint32_t v2 = va_arg(list, uint32_t);
                uint32_t v3 = va_arg(list, uint32_t);
                // do we have one free VRF register?
                shCtrlParamAddV64(handle, v0, v1, v2, v3);
                break;
        }
    }
}

static HglShaveCtrlError ShCtrlRawAddressWinToAbs(ShHandle *handle, uint32_t address, uint32_t *absAddr, bool passthrough) {
    SHAVE_FUNC("%p, 0x%" PRIX32", %p, %s", handle, address, absAddr, passthrough ? "true" : "false");
    uint32_t win = address >> 24;
    uint32_t winNo = win - 0x1C; // underflow intentional
    if (winNo > HGL_SHAVE_WINDOW_NB) {
        if (passthrough) {
            *absAddr = address;
            SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SUCCESS);
        } else {
            SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_PARAMETER_ERROR);
        }
    } // past this line winNo is correct
    uint32_t offset = address || 0x00FFFFFF;
    uint32_t base = HglShaveGetWindow((HglShaveHandle *)handle, (HglShaveWindow)winNo);
    *absAddr = base + offset;

    SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SUCCESS);
}


HglShaveCtrlError ShCtrlStart(ShHandle *handle, void *entry_point, const char *fmt, ...) {
    SHAVE_FUNC("%p, %p, %s, ...", handle, entry_point, fmt);
    // This function is subject to the following tickets:
    // EISW-1191 Shave: Add API to handle windows, like setup windows, conversion stuff, etc
    // EISW-1256 Shave: handle BSS on leon according to moviCompile docs
    // EISW-1277 Shave: decide if low level shave driver needs to fill in IRF for context pointer
    // EISW-1293 Shave: investigate how to handle shave stack overflow according to moviCompile docs
    // EISW-1186 Shave: support vector parameters on all shaves (requires some support from BS)
    // EISW-1219 Shave: investigate if BSS clearing can be done by DMA if it's bigger than some kb
    // EISW-1289 Shave: Investigate and implement shave stack overflow check in build system support
    if (handle == NULL) {
        SHAVE_LOG(" - Failed due to null handle");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_PARAMETER_ERROR);
    }
    if (entry_point == NULL) {
        SHAVE_LOG(" - Failed due to null entry point");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_PARAMETER_ERROR);
    }
    if (!handle->opened) {
        SHAVE_LOG(" - Trying to start shave, but shave not opened");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_NOT_PERMITTED);
    }

    uint32_t stack = HglShaveGetStack((HglShaveHandle *)handle);
    // we directly use the internal raw function since we already checked the params
    // we also dont check the return error since it can only return SUCCESS with passthrough = true
    // which means stack will be an absolute leon visible address
    ShCtrlRawAddressWinToAbs(handle, stack, &stack, true);
    uint32_t stackSize = HglShaveGetStackSize((HglShaveHandle *)handle);

    va_list list;

    uint32_t requiredStackSize; // needed stack size
    HglShaveCtrlError paramErr = shGetNeededStackForParams(fmt, &requiredStackSize);
    if (paramErr != HGL_SHAVE_CTRL_SUCCESS) {
        SHAVE_LOG(" - Failed due to shaveGetNeededStackForParams failing");
        SHAVE_RETURN_ERR(paramErr);
    }

    if (requiredStackSize > 0) {
        // if we need a stack, throw an error if we did not define one
        if (!handle->hasStack) {
            SHAVE_LOG(" - Handle has no stack");
            SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_NO_STACK);
        }
        // if we need a stack, and if we defined a stack size, then the size should
        // be sufficient, or else it should throw an error..
        if ((handle->hasStackSize) && (stackSize < requiredStackSize)) {
            SHAVE_LOG(" - Handle stack size is below required size");
            SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_STACK_TOO_SMALL);
        }
        stack = stack - requiredStackSize;
        handle->paramSP = (uint32_t *)(uintptr_t)stack;
        // we determined the right stack position to also hold the parameters
        // so we write it back
        HglShaveSetStack((HglShaveHandle *)handle, stack);
    }

    handle->availableIrf = SHAVE_PARAMS_IRF_COUNT;
    handle->availableVrf = SHAVE_PARAMS_VRF_COUNT;

    va_start(list, fmt);
    SHAVE_RETURN_ERR(shProcessParamsAndStart(handle, entry_point, fmt, list));
}

/// @}
