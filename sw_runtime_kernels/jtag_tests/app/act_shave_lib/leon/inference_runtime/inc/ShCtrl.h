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
