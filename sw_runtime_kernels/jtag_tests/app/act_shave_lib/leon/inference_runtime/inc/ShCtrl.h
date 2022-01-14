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

#ifndef SHCTRL_H
#define SHCTRL_H

#include <HglShaveCommon.h>
#if defined(SHAVE_PROCESSOR_MAIN)

#include <HglShaveCtrl.h>
#include <HglShaveCtrlErrors.h>
#include <ShCtrlInternal.h>
#include <HglShaveValidityCheck.h>
#include <HglShaveLogging.h>

#endif

#define SHAVE_STACK_ALIGNMENT_MASK 0x07

#ifdef __cplusplus
extern "C" {
#endif

HglShaveCtrlError ShCtrlInit(void);
HglShaveCtrlError ShCtrlOpen(HglShaveType type, uint32_t id, ShHandle **handle);
HglShaveCtrlError ShCtrlClose(ShHandle **handle);
HglShaveCtrlError ShCtrlSetStackAddr(ShHandle *handle, uint32_t stack);
HglShaveCtrlError ShCtrlSetStackSize(ShHandle *handle, uint32_t size);
HglShaveCtrlError ShCtrlSetWindowAddr(ShHandle *handle, HglShaveWindow win, uint32_t winAddr);
HglShaveCtrlError ShCtrlStop(ShHandle *handle);
HglShaveCtrlError ShCtrlStart(ShHandle *handle, void *entry_point, const char *fmt, ...);
uint32_t ShCtrlGetCurrentProcessor(void);
HglShaveCtrlError ShCtrlIsrPrepare(ShHandle *handle);

#ifdef __cplusplus
}
#endif

#endif  //SHCTRL_H
