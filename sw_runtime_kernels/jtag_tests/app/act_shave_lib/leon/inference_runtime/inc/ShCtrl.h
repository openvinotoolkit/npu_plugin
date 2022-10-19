//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#ifndef SHCTRL_H
#define SHCTRL_H

#include <HglShaveCommon.h>
#if defined(SHAVE_PROCESSOR_MAIN)

#include <HglShaveCtrl.h>
#include <HglShaveCtrlRegs.h>
#include <HglShaveCtrlErrors.h>
#include <ShCtrlInternal.h>
#include <HglShaveValidityCheck.h>

#endif

#define SHAVE_STACK_ALIGNMENT_MASK 0x07

#ifdef __cplusplus
extern "C" {
#endif

HglShaveCtrlError ShCtrlInit(void);
HglShaveCtrlError ShCtrlOpen(HglShaveType type, uint32_t id, ShHandle **handle);
HglShaveCtrlError ShCtrlClose(ShHandle **handle);
HglShaveCtrlError ShCtrlSetStackAddr(ShHandle *handle, uintptr_t addr);
HglShaveCtrlError ShCtrlSetStackSize(ShHandle *handle, uint32_t nbytes);
HglShaveCtrlError ShCtrlSetWindowAddr(ShHandle *handle, int winIdx, uintptr_t winAddr);
HglShaveCtrlError ShCtrlStop(ShHandle *handle);
HglShaveCtrlError ShCtrlStart(ShHandle *handle, void *entry_point, const char *fmt, ...);
uint32_t ShCtrlGetCurrentProcessor(void);
HglShaveCtrlError ShCtrlIsrPrepare(ShHandle *handle);

#ifdef __cplusplus
}
#endif

#endif  //SHCTRL_H
