//#include <HglShaveCtrlErrors.h>
//#include <ShaveCtrlInternal.h>
//#include <ShaveCtrlImports.h>



#include <HglShaveCommon.h>
#if defined(SHAVE_PROCESSOR_MAIN)

//#include <stdint.h>
//#include <stdbool.h>
#include <HglShaveCtrl.h>
//#include <ShaveCtrl.h>
#include <HglShaveCtrlErrors.h>
//#include <ShaveCtrlErrors.h>
#include <ShCtrlInternal.h>
//#include <ShaveSharedInternal.h>
#include <HglShaveValidityCheck.h>
#include <HglShaveLogging.h>
//#include <rtems.h>
//#include <sched.h>

#endif


//#if defined(__shave__)
//
//#include <HglShaveId.h>
//
//#define ShaveGetType HglShaveGetType
//
//#define ShaveGetId HglShaveGetId
//
//#endif

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

//uint32_t ShCtrlGetCurrentProcessor(void);
//uint32_t ShCtrlGetIrqLine(ShaveType type, uint32_t id);
#ifdef __cplusplus
}
#endif
