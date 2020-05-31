///
/// INTEL CONFIDENTIAL
/// Copyright 2020. Intel Corporation.
/// This software and the related documents are Intel copyrighted materials, 
/// and your use of them is governed by the express license under which they were provided to you ("License"). 
/// Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or 
/// transmit this software or the related documents without Intel's prior written permission.
/// This software and the related documents are provided as is, with no express or implied warranties, 
/// other than those that are expressly stated in the License.
///
/// @file      osdUtils.h
/// 

#ifndef __OSD_UTILS_H__
#define __OSD_UTILS_H__

#include "osdApi.h"
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t OsduRgb2Yuv  (uint32_t argb);
void     OsduInitBuff (OsdBuff  *b, void *a0, void *a1, void *a2, uint32_t w, uint32_t h, uint32_t s);
void     OsduGenNv12Bg(uint8_t *img, uint32_t imgW, uint32_t imgH);
void     OsduSvuInfo  (uint32_t svu);
void     OsduRandFill (uint32_t seed, uint8_t *data, uint32_t sz);
void     OsduTestInit ();

//Polygon helpers
void     OsduPolyTranslate(OsdPoint *verts, uint32_t nVerts, OsdPoint shift);
void     OsduPolyScale    (OsdPoint *verts, uint32_t nVerts, float scale);
void     OsduPolyRotate   (OsdPoint *verts, uint32_t nVerts, float angle/*deg*/);


//Bmp helpers
void OsduScaleL   (uint8_t *iL, uint32_t iW, uint32_t iH,
                   uint8_t *oL, uint32_t oW, uint32_t oH);

void OsduScaleC   (uint8_t *iC,  uint32_t iW, uint32_t iH,
                   uint8_t *oC,  uint32_t oW, uint32_t oH);

void OsduScaleNV12(uint8_t *inp, uint32_t iW, uint32_t iH,
                   uint8_t *out, uint32_t oW, uint32_t oH);

uint32_t countNzPix(uint8_t *buff, uint32_t sz);

#define ARGB(a,r,g,b) (((a)<<24)|((r)<<16)|((g)<<8)|(b))

static inline void OsduLL2Cflush(const void *addr, size_t size){
    // assert( (((uint32_t)addr) & 63) == 0);
    // assert( (((uint32_t)size) & 63) == 0);
    // Do nothing. - TODO remove
}

static inline void OsduLL2Cinval(const void *addr, size_t size){
    // assert( (((uint32_t)addr) & 63) == 0);
    // assert( (((uint32_t)size) & 63) == 0);
    // Do nothing. - TODO remove
}

#ifdef __cplusplus
} //extern "C"
#endif

#endif
