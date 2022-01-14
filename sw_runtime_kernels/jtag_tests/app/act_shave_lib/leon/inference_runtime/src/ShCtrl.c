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

#define CONFIG_NN_LOG_VERBOSITY_LRT_INFO
#include <mv_types.h>
#include <OsDrvBootShave.h>
#include <ShaveL2Cache.h>
#include <ShCtrl.h>

#if __has_include(<bsp/irq.h>)
#include <bsp/irq.h>
#endif
#if __has_include(<bsp.h>)
#include <bsp.h>
#endif

#define LOCAL_SH_COUNT (HGL_NCE_ACT_SHAVE_NB + HGL_NCE_DPU_NB)

ShHandle ShConfig[LOCAL_SH_COUNT];

// This file has potentially HW dependent functions
// No other file should be changed depending on VPU version

// Reason on HW dependency:
//  on VPU2.7 RST register exists
//  on VPU4   RST register does not - CPR needs to be used
static void ShCtrlResetShaveHw(ShHandle *handle) {
    SHAVE_FUNC("%p", handle);
    // we blindly release the handle since we reset the hardware
    SHAVE_LOG(" - Beginning resetting shave HW");
    SHAVE_LOG(" - Releasing semaphore for handle %p", handle);
    rtems_semaphore_release(handle->waitSema);
#if defined(SHAVE_HAS_RESET)
    HglShaveReset((HglShaveHandle *)handle);
#else
    SHAVE_LOG(" - Nothing to do. Shave RST not supported");
    SHAVE_LOG(" - Consider using CPR for reset");
#endif
    SHAVE_LOG(" - Shave HW reset done");
}

uint32_t ShCtrlGetCurrentProcessor(void) {
#if defined(SHAVE_PLATFORM_37xx)
    return rtems_get_current_processor();
#elif defined(SHAVE_PLATFORM_40xx)
    return rtems_scheduler_get_processor();
#endif
}

#if defined(SHAVE_PROCESSOR_MAIN)
#if defined(SHAVE_PLATFORM_37xx)

static uint32_t ShCtrlGetIrqLine(HglShaveType type, uint32_t id) {
    SHAVE_FUNC("%" PRId32", %" PRId32"", type, id);
    uint32_t irq[] = {LRT_IRQ_SHAVE, LNN_IRQ_DPU_IRQ_0_0, LNN_IRQ_DPU_IRQ_16_0};
    uint32_t line = irq[type];
    if (type == SHAVE_UPA) {
        SHAVE_RETURN(line + id, "%" PRId32"");
    } else {
        SHAVE_RETURN(line + id * 2, "%" PRId32"");
    }
}

static uint32_t ShCtrlGetHandleIndex(HglShaveType type, uint32_t id) {
    SHAVE_FUNC("%" PRId32", %" PRId32"", type, id);
#if defined(__leon_rt__)
    (void)type; // unused in leon_rt
    SHAVE_RETURN(id, "%" PRId32"");
#endif
#if defined(__leon_nn__)
    SHAVE_RETURN(((type == SHAVE_ACT) * HGL_MAX_ACT_SHAVES) + id, "%" PRId32"");
#endif
}

#endif
#endif

#if defined(SHAVE_PROCESSOR_MAIN)

#define IRQ_PRIO 10

RTEMS_INTERRUPT_LOCK_DEFINE(static, shv_lock, "SHAVE_LOCK");

static void ShCtrlIrqHandler(void *arg) {
    SHAVE_FUNC("%p", arg);
    rtems_interrupt_lock_context context;
    rtems_interrupt_lock_acquire(&shv_lock, &context);

    ShHandle *handle = (ShHandle *)arg;
    rtems_semaphore_release(handle->waitSema);

    rtems_interrupt_lock_release(&shv_lock, &context);
}

HglShaveCtrlError ShCtrlIsrPrepare(ShHandle *handle) {
    SHAVE_FUNC("%p", handle);
    const uint32_t irq = ShCtrlGetIrqLine(handle->type, handle->id);
    const uint32_t proc = ShCtrlGetCurrentProcessor();
    BSP_Clear_interrupt(irq);
    BSP_Set_interrupt(irq, POS_EDGE_INT, IRQ_PRIO, proc);
    HglShaveClearAllIrqs((HglShaveHandle *)handle);
    SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SUCCESS);
}

static HglShaveCtrlError ShCtrlInstallISR(ShHandle *handle) {
    SHAVE_FUNC("%p", handle);
    rtems_status_code ret;
    const uint32_t irq = ShCtrlGetIrqLine(handle->type, handle->id);
    ShCtrlIsrPrepare(handle);

    // Register the rtems interrupt handle for this shave.

    ret = BSP_interrupt_register(irq, NULL, ShCtrlIrqHandler, (void *)handle);
    if (ret != RTEMS_SUCCESSFUL) {
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_ISR_FAILURE);
    }

    HglShaveSetIsrSgi((HglShaveHandle *)handle, true);
    SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SUCCESS);
}

static HglShaveCtrlError ShCtrlRemoveISR(ShHandle *handle) {
    SHAVE_FUNC("%p", handle);
    rtems_status_code ret;
    ret = BSP_interrupt_unregister(ShCtrlGetIrqLine(handle->type, handle->id), ShCtrlIrqHandler, (void *)handle);
    SHAVE_RETURN_ERR(ret != RTEMS_SUCCESSFUL ? HGL_SHAVE_CTRL_ISR_FAILURE : HGL_SHAVE_CTRL_SUCCESS);
}

#endif

static void SetShHandle(HglShaveType type, uint32_t id, ShHandle *handle) {
    SHAVE_FUNC("%" PRId32", %" PRId32", %p", type, id, handle);
    memset(handle, 0, sizeof(HglShaveType));
    HglShaveSetHandle(type, id, (HglShaveHandle *)handle);
}

static void ResetShHandle(ShHandle *handle) {
    SHAVE_FUNC("%p", handle);
    // resetBegin and resetEnd are 1 byte types (uint8_t)
    const size_t resetSize = handle->resetEnd - handle->resetBegin;
    memset(&handle->resetBegin, 0, resetSize);
}

static void shSemaCleanup(void) {
    SHAVE_FUNC("");
    for (uint32_t s = 0; s < LOCAL_SH_COUNT; s++) {
        ShHandle *h = &ShConfig[s];
        // ignoring return value as we just try to clean as much
        // as we can.. we are in a error state anyway
        rtems_semaphore_delete(h->waitSema);
    }
}

static bool shHandleCreate(HglShaveType type) {
    SHAVE_FUNC("%" PRId32"", type);
    rtems_status_code rs;
    static uint32_t semaId = 0;
    uint32_t count = HglShaveMaxId[type];
    for (uint32_t id = 0; id < count; id++) {
        SHAVE_LOG(" - Creating sema %" PRId32"", id);
        uint32_t idx = ShCtrlGetHandleIndex(type, id);
        ShHandle *h = &ShConfig[idx];
        SetShHandle(type, id, h);
        rs = rtems_semaphore_create(rtems_build_name('S', 'H', 'S', '0' + semaId++), 0, RTEMS_SIMPLE_BINARY_SEMAPHORE,
                                    0, &h->waitSema);

        if (RTEMS_SUCCESSFUL != rs) {
            semaId = 0;
            SHAVE_LOG(" - !!! Starting cleanup");
            shSemaCleanup();

            SHAVE_RETURN(true, "ERROR"); // true means error
        }
    }
    SHAVE_RETURN(false, "SUCCESS"); // false means success
}

HglShaveCtrlError ShCtrlInit(void) {
    SHAVE_FUNC("");
    static bool initDone = false;
    SHAVE_LOG(" - Init %s", initDone ? "was already done" : "was not previously done");
    if (!initDone) {
        SHAVE_LOG(" - Starting sema init");
        if (RTEMS_SUCCESSFUL != ShaveSemaphoresInit()) {
            SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SEMA_FAILURE);
        }

#if LOCAL_SH_COUNT > 0
        for (uint32_t type = 0; type < HGL_SHAVE_TYPE_NB; type++) {
            if (HglShaveAccessAllowed[type]) {
                SHAVE_LOG(" - Creating shave semaphores for %s", typeCharptr[type]);
                if (shHandleCreate((HglShaveType)type)) {
                    SHAVE_LOG(" - Failed to create semaphores");
                    SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SEMA_FAILURE);
                }
            }
        }
        initDone = true;
        SHAVE_LOG(" - Initialization marked as done");
#else

        SHAVE_LOG(" - Nothing to do, no local shaves");
#endif
    }
    SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SUCCESS);
}

static inline HglShaveCtrlError protect(ShHandle *handle) {
    SHAVE_FUNC("%p", handle);
    HglShaveCtrlError err = (HglShaveCtrlError)ShaveSvuLock(handle->type, handle->id);
    SHAVE_RETURN_ERR(err);
}

static inline HglShaveCtrlError unprotect(ShHandle *handle) {
    SHAVE_FUNC("%p", handle);
    HglShaveCtrlError err = (HglShaveCtrlError)ShaveSvuUnlock(handle->type, handle->id);
    SHAVE_RETURN_ERR(err);
}

HglShaveCtrlError ShCtrlOpen(HglShaveType type, uint32_t id, ShHandle **handle) {
    SHAVE_FUNC("%" PRId32", %" PRId32", %p", type, id, handle);
    if (handle == NULL) {
        SHAVE_LOG(" - Shave *handle null");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_PARAMETER_ERROR);
    }

    SHAVE_LOG(" - Will write handle to address %p", handle);
    HglShaveGeneralError hsgerr = HglShaveIsTypeIdValid(type, id);
    if (HGL_SHAVE_SUCCESS != hsgerr) {
        SHAVE_LOG(" - Shave handle is not valid");
        SHAVE_RETURN_ERR((HglShaveCtrlError)hsgerr);
    }

    SHAVE_LOG(" - Opening [%s, %" PRId32"]", typeCharptr[type], id);
    uint32_t idx = ShCtrlGetHandleIndex(type, id);
    SHAVE_LOG(" - Using handle id %" PRId32"", idx);
    ShHandle *h = &ShConfig[idx];
    SHAVE_LOG(" - Using handle %p", h);
    if (h->opened) {
        SHAVE_LOG(" - Handle already opened");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_NOT_PERMITTED);
    } else {
        *handle = h;
        SHAVE_LOG(" - Protecting handle");
        HglShaveCtrlError er = protect(h);
        SHAVE_LOG(" - Protecting handle %" PRId32"", er);
        if (er != HGL_SHAVE_CTRL_SUCCESS) {
            // doesnt need unprotect
            // since protect failed
            SHAVE_LOG(" - Protecting handle failed");
            SHAVE_RETURN_ERR(er);
        }
        SHAVE_LOG(" - Marking handle %p as opened", handle);
        h->opened = true;
        ShCtrlResetShaveHw(h);
#ifndef SHAVE_WAIT_POLL
        SHAVE_LOG(" - Installing ISR");
        er = ShCtrlInstallISR(h);
        SHAVE_LOG(" - Installing ISR status %" PRId32"", er);
#endif
        HglShaveCtrlError unprotErr = unprotect(h);
        SHAVE_LOG(" - Unprotecting");
        if (unprotErr != HGL_SHAVE_CTRL_SUCCESS) {
            SHAVE_LOG(" - Unprotecting failed");
            SHAVE_RETURN_ERR(unprotErr);
        }
        SHAVE_RETURN_ERR(er);
    }
}

HglShaveCtrlError ShCtrlClose(ShHandle **handle) {
    SHAVE_FUNC("%p", handle);
    HglShaveCtrlError er;
    ShHandle *hCopy;
    if (handle == NULL || *handle == NULL) {
        SHAVE_LOG(" - Failed due to null handle");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_PARAMETER_ERROR);
    }
    // this function should be protected since it potentially modifies the handle
    er = protect(*handle);
    if (er != HGL_SHAVE_CTRL_SUCCESS) {
        SHAVE_LOG(" - Failed due to protect failing");
        SHAVE_RETURN_ERR(er);
    }
    if (!(*handle)->opened) {
        // We dont check unprotect since it's way more important to
        // return NOT_PERMITTED anyway, but we try to unprotect
        unprotect(*handle);
        SHAVE_LOG(" - Trying to close shave, but shave not opened");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_NOT_PERMITTED);
    } else {
        (*handle)->opened = false;
#ifndef SHAVE_WAIT_POLL
        er = ShCtrlRemoveISR(*handle);
        if (er != HGL_SHAVE_CTRL_SUCCESS) {
            unprotect(*handle);
            SHAVE_LOG(" - Failed due to ShCtrlRemoveISR");
            SHAVE_RETURN_ERR(er);
        }
#endif
    }
    hCopy = *handle;
    *handle = NULL;
    ResetShHandle(hCopy);
    // We unprotect before exiting since the scope of protect-unprotect is to only
    // protect at function level only, while handle->opened will protect at API level
    er = unprotect(hCopy);
    if (er != HGL_SHAVE_CTRL_SUCCESS) {
        SHAVE_LOG(" - Failed due to unprotect failing");
        SHAVE_RETURN_ERR(er);
    }

    SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SUCCESS);
}

HglShaveCtrlError ShCtrlSetStackAddr(ShHandle *handle, uint32_t stack) {
    SHAVE_FUNC("%p, 0x%" PRIX32"", handle, stack);
    if (handle == NULL) {
        SHAVE_LOG(" - Failed due to null handle");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_PARAMETER_ERROR);
    }
    if (!handle->opened) {
        SHAVE_LOG(" - Trying to set stack address, but shave not opened");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_NOT_PERMITTED);
    }

    if ((stack & SHAVE_STACK_ALIGNMENT_MASK) > 0) {
        SHAVE_LOG(" - stack not aligned");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_STACK_NOT_ALIGNED);
    }

    handle->hasStack = true;
    HglShaveSetStack((HglShaveHandle *)handle, stack);
    SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SUCCESS);
}

HglShaveCtrlError ShCtrlSetStackSize(ShHandle *handle, uint32_t size) {
    SHAVE_FUNC("%p, %" PRId32"", handle, size);
    if (handle == NULL) {
        SHAVE_LOG(" - Failed due to null handle");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_PARAMETER_ERROR);
    }
    if (!handle->opened) {
        SHAVE_LOG(" - Trying to set stack size, but shave not opened");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_NOT_PERMITTED);
    }
    handle->hasStackSize = true;
    HglShaveSetStackSize((HglShaveHandle *)handle, size);
    SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SUCCESS);
}

HglShaveCtrlError ShCtrlSetWindowAddr(ShHandle *handle, HglShaveWindow win, uint32_t winAddr) {
    SHAVE_FUNC("%p, %" PRId32", %p", handle, win, winAddr);
    if (handle == NULL) {
        SHAVE_LOG(" - Failed due to null handle");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_PARAMETER_ERROR);
    }
    if (!handle->opened) {
        SHAVE_LOG(" - Trying to set window address, but shave not opened");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_NOT_PERMITTED);
    }
    if (win >= SHAVE_WINDOW_NB) {
        SHAVE_LOG(" - Failed due to ShaveWindow being greater then SHAVE_WINDOW_NB");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_PARAMETER_ERROR);
    }
    HglShaveSetWindow((HglShaveHandle *)handle, win, winAddr);
    SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SUCCESS);
}

HglShaveCtrlError ShCtrlStop(ShHandle *handle) {
    SHAVE_FUNC("%p", handle);
    if (handle == NULL) {
        SHAVE_LOG(" - Failed due to null handle");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_PARAMETER_ERROR);
    }
    if (!handle->opened) {
        SHAVE_LOG(" - Trying to stop shave, but shave not opened");
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_NOT_PERMITTED);
    }

    HglShaveStop((HglShaveHandle *)handle);

    rtems_status_code sc = rtems_semaphore_release(handle->waitSema);
    if (RTEMS_SUCCESSFUL != sc) {
        SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SEMA_FAILURE);
    }

    SHAVE_RETURN_ERR(HGL_SHAVE_CTRL_SUCCESS);
}
