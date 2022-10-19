//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#define CONFIG_NN_LOG_VERBOSITY_LRT_INFO
#include <mv_types.h>
#include <OsDrvBootShave.h>
#include <ShaveL2Cache.h>
#include <ShCtrl.h>
#include <mvLog.h>

#if __has_include(<bsp/irq.h>)
#include <bsp/irq.h>
#endif
#if __has_include(<bsp.h>)
#include <bsp.h>
#endif

#define SH_ADDR_STACK_ALIGN_MASK (0x07)

uint32_t ShCtrlGetCurrentProcessor(void) {
#if defined(SHAVE_PLATFORM_37xx)
    return rtems_get_current_processor();
#elif defined(SHAVE_PLATFORM_40xx)
    return rtems_scheduler_get_processor();
#endif
}

#define LOCAL_SH_COUNT (HGL_NCE_ACT_SHAVE_NB + HGL_NCE_DPU_NB)
ShHandle ShConfig[LOCAL_SH_COUNT];

#if defined(SHAVE_PROCESSOR_MAIN)
#if defined(SHAVE_PLATFORM_37xx)

static uint32_t ShCtrlGetIrqLine(HglShaveType type, uint32_t id) {
    uint32_t irq[] = {LRT_IRQ_SHAVE, LNN_IRQ_DPU_IRQ_0_0, LNN_IRQ_DPU_IRQ_16_0};
    uint32_t line = irq[type];
    if (type == SHAVE_UPA) {
        return line + id;
    } else {
        return line + id * 2;
    }
}

static uint32_t ShCtrlGetHandleIndex(HglShaveType type, uint32_t id) {
#if defined(__leon_rt__)
    (void)type; // unused in leon_rt
    return id;
#endif
#if defined(__leon_nn__)
    return ((type == SHAVE_ACT) * HGL_MAX_ACT_SHAVES) + id;
#endif
}

#endif
#endif

#if defined(SHAVE_PROCESSOR_MAIN)

#define IRQ_PRIO 10

static void ShCtrlIrqHandler(void *arg) {
    ShHandle *handle = (ShHandle *)arg;
    rtems_semaphore_release(handle->waitSema);
}

static void shaveCtrlInterruptHandler(void *arg) {
    ShHandle *handle = (ShHandle *)arg;
    rtems_semaphore_release(handle->waitSema);
}

static void disableShaveInterrupts(uint8_t *base) {
    ShaveOcrReg ocr = {.s1g0 = 1};
    SET_REG_WORD(base + LSHV_OCR_OFFSET, ocr.whole); // Stop the Shave
    SET_REG_WORD(base + LSHV_ICR_OFFSET, 0);         // Disable all interrupts
    SET_REG_WORD(base + LSHV_IRR_OFFSET, 0x3F);      // Clear all interrupt flags
}

static HglShaveCtrlError ShCtrlInstallISR(ShHandle *handle) {
    rtems_status_code sc;

    // The Shave is allowed to be powered off at this point in time. If this
    // is the case, writes will have no effect and reads will return zero.
    disableShaveInterrupts(handle->base);

    // Get IRQ information
    const uint32_t irq_line = ShCtrlGetIrqLine(handle->type, handle->id);
    uint32_t irq_affinity = ShCtrlGetCurrentProcessor();
    uint32_t irq_priority = 10;

    BSP_Clear_interrupt(irq_line);
    BSP_Set_interrupt(irq_line, POS_EDGE_INT, irq_priority, irq_affinity);
    sc = BSP_interrupt_register(irq_line, NULL, shaveCtrlInterruptHandler, handle);
    if (sc != RTEMS_SUCCESSFUL) {
        mvLog(MVLOG_ERROR, "rtems_interrupt_handler_install failed: %s", rtems_status_text(sc));
        return HGL_SHAVE_CTRL_ISR_FAILURE;
    }

    return HGL_SHAVE_CTRL_SUCCESS;
}

static HglShaveCtrlError ShCtrlRemoveISR(ShHandle *handle) {
    rtems_status_code sc;

    // As with shCtrlInstallISR, stop the shave and disable its interrupts
    disableShaveInterrupts(handle->base);

    // Remove interrupt handler
    uint32_t irq_line = ShCtrlGetIrqLine(handle->type, handle->id);
    BSP_Clear_interrupt(irq_line);
    sc = BSP_interrupt_unregister(irq_line, ShCtrlIrqHandler, handle);

    if (sc != RTEMS_SUCCESSFUL) {
        mvLog(MVLOG_ERROR, "rtems_interrupt_handler_remove failed: %s", rtems_status_text(sc));
        return HGL_SHAVE_CTRL_ISR_FAILURE;
    }

    return HGL_SHAVE_CTRL_SUCCESS;
}

#endif

static rtems_id shOpenedSema;

static HglShaveCtrlError shCtrlCreateOpenedSemaphore(void) {
    rtems_status_code sc;

    if (shOpenedSema) {
        mvLog(MVLOG_ERROR, "Shave Open Semaphore was already created");
        return HGL_SHAVE_CTRL_NOT_PERMITTED;
    }

    sc = rtems_semaphore_create(rtems_build_name('S', 'H', 'V', 'O'), 1,
                                RTEMS_BINARY_SEMAPHORE | RTEMS_PRIORITY | RTEMS_INHERIT_PRIORITY, 0, &shOpenedSema);
    if (sc != RTEMS_SUCCESSFUL) {
        mvLog(MVLOG_ERROR, "Creating open semaphore failed with status: %s", rtems_status_text(sc));
        return HGL_SHAVE_CTRL_SEMA_FAILURE;
    }

    return HGL_SHAVE_CTRL_SUCCESS;
}

static void SetShHandle(HglShaveType type, uint32_t id, ShHandle *handle) {
    memset(handle, 0, sizeof(HglShaveType));

    handle->base = HglGetShaveBaseAddr(type, id);
    handle->type = type;
    handle->id = id;
}

static void ResetShHandle(ShHandle *handle) {
    // Copy persistent members out of the Shave handle structure
    uint8_t *base = handle->base;
    ShaveType type = handle->type;
    uint32_t id = handle->id;
    rtems_id waitSema = handle->waitSema;

    // Wipe the handle clean; then copy the persistent members back
    memset(handle, 0, sizeof(ShHandle));

    handle->base = base;
    handle->type = type;
    handle->id = id;
    handle->waitSema = waitSema;
}

static void shSemaCleanup(void) {
    for (uint32_t s = 0; s < LOCAL_SH_COUNT; s++) {
        ShHandle *h = &ShConfig[s];
        // ignoring return value as we just try to clean as much
        // as we can.. we are in a error state anyway
        rtems_semaphore_delete(h->waitSema);
    }
}

static bool shHandleCreate(HglShaveType type) {
    rtems_status_code rs;
    static uint32_t semaId = 0;
    uint32_t count = HglShaveMaxId[type];
    for (uint32_t id = 0; id < count; id++) {
        uint32_t idx = ShCtrlGetHandleIndex(type, id);
        ShHandle *h = &ShConfig[idx];
        SetShHandle(type, id, h);
        rs = rtems_semaphore_create(rtems_build_name('S', 'H', 'S', '0' + semaId++), 1, RTEMS_SIMPLE_BINARY_SEMAPHORE,
                                    0, &h->waitSema);

        if (RTEMS_SUCCESSFUL != rs) {
            mvLog(MVLOG_ERROR, "Creating semaphore failed with status: %s", rtems_status_text(rs));
            semaId = 0;
            shSemaCleanup();

            return true; // true means error
        }
    }
    return false; // false means success
}

HglShaveCtrlError ShCtrlInit(void) {
    static bool initDone = false;

    if (initDone) {
        mvLog(MVLOG_WARN, "ShaveCtrl already initialised");
    } else {
        if (RTEMS_SUCCESSFUL != ShaveSemaphoresInit()) {
            mvLog(MVLOG_ERROR, "ShaveSemaphoresInit failed");
            return HGL_SHAVE_CTRL_SEMA_FAILURE;
        }

        if (shCtrlCreateOpenedSemaphore() != HGL_SHAVE_CTRL_SUCCESS)
            return HGL_SHAVE_CTRL_SEMA_FAILURE;

#if LOCAL_SH_COUNT > 0
        for (uint32_t type = 0; type < HGL_SHAVE_TYPE_NB; type++) {
            if (HglShaveAccessAllowed[type]) {
                if (shHandleCreate((ShaveType)type)) {
                    mvLog(MVLOG_ERROR, "shaveHandleCreate failed");
                    return HGL_SHAVE_CTRL_SEMA_FAILURE;
                }
            }
        }
        initDone = true;
#else
        mvLog(MVLOG_WARN, "ShaveCtrlInit: no local shaves");
#endif
    }

    return HGL_SHAVE_CTRL_SUCCESS;
}

HglShaveCtrlError ShCtrlOpen(HglShaveType type, uint32_t id, ShHandle **handle) {
    rtems_status_code sc;
    ShHandle *hnd;
    uint32_t hidx;

    if (!handle) {
        mvLog(MVLOG_ERROR, "Null handle pointer");
        return HGL_SHAVE_CTRL_PARAMETER_ERROR;
    }

    HglShaveGeneralError hsgerr = HglShaveIsTypeIdValid(type, id);
    if (HGL_SHAVE_SUCCESS != hsgerr) {
        mvLog(MVLOG_ERROR, "Shave handle is not valid");
        return (HglShaveCtrlError)hsgerr;
    }

    hidx = ShCtrlGetHandleIndex(type, id);
    mvLog(MVLOG_DEBUG, "ShaveCtrlOpen: Using handle idx %" PRIu32, hidx);
    hnd = &ShConfig[hidx];

    sc = rtems_semaphore_obtain(shOpenedSema, RTEMS_WAIT, RTEMS_NO_TIMEOUT);
    if (sc != RTEMS_SUCCESSFUL) {
        mvLog(MVLOG_ERROR, "rtems_semaphore_obtain failed: %s", rtems_status_text(sc));
        return HGL_SHAVE_CTRL_SEMA_FAILURE;
    }

    if (hnd->opened) {
        rtems_semaphore_release(shOpenedSema);
        mvLog(MVLOG_ERROR, "Shave already opened");
        return HGL_SHAVE_CTRL_NOT_PERMITTED;
    }

    hnd->opened = true;
    rtems_semaphore_release(shOpenedSema);

    // Obtain the wait semaphore at this point rather than wasting time later
    sc = rtems_semaphore_obtain(hnd->waitSema, RTEMS_NO_WAIT, 0);
    if (sc != RTEMS_SUCCESSFUL) {
        mvLog(MVLOG_ERROR, "rtems_semaphore_obtain failed: %s", rtems_status_text(sc));
        hnd->opened = false;
        return HGL_SHAVE_CTRL_SEMA_FAILURE;
    }

    if (ShCtrlInstallISR(hnd) != HGL_SHAVE_CTRL_SUCCESS) {
        rtems_semaphore_release(hnd->waitSema);
        hnd->opened = false;
        return HGL_SHAVE_CTRL_ISR_FAILURE;
    }

    *handle = hnd;
    return HGL_SHAVE_CTRL_SUCCESS;
}

HglShaveCtrlError ShCtrlClose(ShHandle **handle) {
    rtems_status_code sc;
    ShHandle *hnd;

    if (!handle || !*handle) {
        mvLog(MVLOG_ERROR, "Null handle");
        return HGL_SHAVE_CTRL_PARAMETER_ERROR;
    }
    hnd = *handle;

    if (!hnd->opened) {
        mvLog(MVLOG_ERROR, "Shave not opened");
        return HGL_SHAVE_CTRL_NOT_PERMITTED;
    }

    ShCtrlRemoveISR(hnd);
    rtems_semaphore_release(hnd->waitSema);
    HglShaveReset(hnd->base);

    sc = rtems_semaphore_obtain(shOpenedSema, RTEMS_WAIT, RTEMS_NO_TIMEOUT);
    if (sc != RTEMS_SUCCESSFUL) {
        mvLog(MVLOG_ERROR, "rtems_semaphore_obtain failed: %s", rtems_status_text(sc));
        return HGL_SHAVE_CTRL_SEMA_FAILURE;
    }

    ResetShHandle(hnd);
    *handle = NULL;
    rtems_semaphore_release(shOpenedSema);

    return HGL_SHAVE_CTRL_SUCCESS;
}

HglShaveCtrlError ShCtrlSetStackAddr(ShHandle *handle, uintptr_t addr) {
    if (!handle) {
        mvLog(MVLOG_ERROR, "Null handle");
        return HGL_SHAVE_CTRL_PARAMETER_ERROR;
    }

    // An 8 byte alignment is required by the hardware
    addr &= ~SH_ADDR_STACK_ALIGN_MASK;

    handle->stackAddr = addr; // Set the stack pointer
    handle->stackSize = 0;    // Reset the stack size to "not set"
    return HGL_SHAVE_CTRL_SUCCESS;
}

HglShaveCtrlError ShCtrlSetStackSize(ShHandle *handle, uint32_t nbytes) {
    if (!handle) {
        mvLog(MVLOG_ERROR, "Null handle");
        return HGL_SHAVE_CTRL_PARAMETER_ERROR;
    }
    if (!handle->stackAddr) {
        mvLog(MVLOG_ERROR, "Stack address must be set before the stack size");
        return HGL_SHAVE_CTRL_NOT_AVAILABLE;
    }

    // The resulting low-water stack address must be 8 byte aligned
    nbytes &= ~SH_ADDR_STACK_ALIGN_MASK;

    handle->stackSize = nbytes;
    return HGL_SHAVE_CTRL_SUCCESS;
}

HglShaveCtrlError ShCtrlSetWindowAddr(ShHandle *handle, int winIdx, uintptr_t winAddr) {
    if (!handle || winIdx < 0 || winIdx >= SHAVE_WINDOW_NB) {
        mvLog(MVLOG_ERROR, "Bad parameter");
        return HGL_SHAVE_CTRL_PARAMETER_ERROR;
    }
    if (winAddr & SH_ADDR_WINDOW_ALIGN_MASK) {
        mvLog(MVLOG_ERROR, "Window address must be 10 bit aligned");
        return HGL_SHAVE_CTRL_PARAMETER_ERROR;
    }
    handle->winOffset[winIdx] = winAddr;
    return HGL_SHAVE_CTRL_SUCCESS;
}

HglShaveCtrlError ShCtrlStop(ShHandle *handle) {
    if (!handle) {
        mvLog(MVLOG_ERROR, "Null handle");
        return HGL_SHAVE_CTRL_PARAMETER_ERROR;
    }
    if (!handle->opened) {
        mvLog(MVLOG_ERROR, "Shave not opened");
        return HGL_SHAVE_CTRL_NOT_PERMITTED;
    }

    HglShaveStop(handle->base);

    // It's possible that a pending irq remains after the Shave stops
    // Clear this to avoid race conditions with release/obtain waitSema
    uint32_t irq_line = ShCtrlGetIrqLine(handle->type, handle->id);
    BSP_Clear_interrupt(irq_line);

    // No way to know if IRQ fired or not; attempt the obtain anyway
    rtems_semaphore_obtain(handle->waitSema, RTEMS_NO_WAIT, 0);
    return HGL_SHAVE_CTRL_SUCCESS;
}
