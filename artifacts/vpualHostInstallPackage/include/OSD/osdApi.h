#ifndef __OSD_API_H__
#define __OSD_API_H__

#include "osdApiDefs.h"

#ifdef __cplusplus
extern "C" {
#endif

void OsdListInit(OsdList *lst, OsdAddr  base, uint32_t size);

void OsdAlphaBmp(OsdList *lst, OsdBmpDesc  *d);
void OsdAlphaCol(OsdList *lst, OsdMaskDesc *d);
void OsdPoly    (OsdList *lst, OsdPolyDesc *d);
void OsdMosaic  (OsdList *lst, OsdMosDesc  *d);
void OsdFrame   (OsdList *lst, OsdFrmDesc  *d);
void OsdBox     (OsdList *lst, OsdBoxDesc  *d);

void OsdDraw    (OsdList *lst, OsdBuff *dest, uint32_t svuNo);
void OsdWait    (uint32_t svuNo);

uint32_t OsdCheck(OsdList *lst, OsdBuff *dest);

//could have multiple instances, svuNo is the "handle"
uint32_t OsdOpen (uint32_t svuNo);
uint32_t OsdClose(uint32_t svuNo);
uint32_t OsdFlag (uint32_t svuNo, uint32_t flag, uint32_t val);

#ifdef __cplusplus
} //extern "C"
#endif

#endif