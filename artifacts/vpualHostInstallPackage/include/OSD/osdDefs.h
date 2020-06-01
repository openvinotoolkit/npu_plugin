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
/// @file      osdDefs.h
/// 

#ifndef __OSD_DEFS_H__
#define __OSD_DEFS_H__

#include "osdApiDefs.h"

#define ALIGNED(x) __attribute__ ((aligned(x)))

//Private defs
enum {
    ID_ALPHA_BMP = 0,  //bitmap alpha blend
    ID_ALPHA_COL,  //solid-color alpha blend
    ID_BOX,        //opaque box
    ID_FRAME,      //frame
    ID_POLYGON,    //polygon mask
    ID_MOSAIC_1,   //privacy mosaic
    ID_MAX
};

//Must match exactly above order !!!
#define THE_PRIMITIVE_NAMES "bmp", "txt", "box", "frm", "ply", "mos"

//Dummy decorators to clarify number of DMA descriptors
//required per primitive/case basis
#define SRC_Y 1 //source-luma
#define SRC_A 1 //source-alpha
#define SRC_C 1 //source-chroma
#define DST_Y 1 //destination-luma
#define DST_C 1 //destination-chroma

//Decorators
#define W(x) x //width
#define H(x) x //height
#define S(x) x //stride
#define SRC(x) x //source
#define DST(x) x //dest

//Buffer factors
#define YF 1.0f //luma
#define AF 1.0f //alpha (for Y)
#define CF 0.5f //chroma (relative to Y)

//End-Of-List marker
#define LIST_END 0xAABBCCDD

//Toggle set no
#define TOGGLE(x) (1-x)

//Need to round things
#define EVEN(x) ((x)&(~1))

//Round up div result
#define DIV_UP(h,l) ((h) + ((l)-1))/(l)

//Round up 'x' to 'T'arget
#define ROUND(x,T) (((x)+(T-1)) & (~(T-1)))

//Round up to Leon L2 cache line size multiple
#define LL2CM(x) ROUND(x,64)

//Round up 'x' to even
#define ROUND_EVEN(x) ROUND(x,2)

//Fractional part
#define FRAC(x) (x - (int)x)

//Round down to arbitrary 't' target
#define ROUND_TO(x,t) ((x/t)*t)

//Number of elements in vector
#define NUM(v) (sizeof(v)/sizeof(v[0]))

//Max resolution
#ifndef OSD_MAX_H
#define OSD_MAX_H 2160 //4K
#endif

#ifndef OSD_MAX_W
#define OSD_MAX_W 3840 //4K
#endif

#define MAX_W OSD_MAX_W //TBD: keep only OSD_MAX_...
#define MAX_H OSD_MAX_H

//Allocation pad (between buffs)
#ifndef APAD
#define APAD 32
#endif

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//CMX mem map (precomputed)
typedef struct {
    uint32_t size;
    OsdBuff  src; //A,Y,C
    OsdBuff  dst; //  Y,C
}mmBmp; //Bitmap

typedef struct {
    uint32_t size;
    OsdBuff  src; //A
    OsdBuff  dst; //Y,C
}mmTxt; //Text

//Do RD only if transparent
typedef struct {
    uint32_t size;
    OsdBuff  dst; //Y,C
}mmBox; //Box

typedef struct {
    uint32_t size;
    OsdBuff  dst; //Y,C
}mmMos; //Mosaic

typedef struct {
    uint32_t size;
    OsdBuff  src; //A
    OsdBuff  dst; //Y,C
}mmFrm; //Frame (empty rectangle)

        struct PolyAuxBufs;
typedef struct PolyAuxBufs PolyAuxBufs;

typedef struct {
    uint32_t     size;
    PolyAuxBufs *aux;
    OsdBuff      dst[2];
}mmPly;

//Main CMX working memory (pixel + other)
#ifndef W_MEM_SZ
#define W_MEM_SZ (2*40*1024)
#endif

//Limits
#define BMP_MIN_W 8
#define BMP_MIN_H 8
#define BOX_MIN_W 4
#define BOX_MIN_H 4
#define MSK_MIN_W 4 //???
#define MSK_MIN_H 4 //???
#define FRM_MIN_W 4 //inner space
#define FRM_MIN_H 4
#define FRM_MIN_T 2
#define FRM_MAX_T 128
#define MOS_MIN_CSZ   2
#define MOS_MAX_CSZ 128

#if defined(OSD_EN_PRINT)
 #define OSD_PRINT printf
#else
 #define OSD_PRINT(...)
#endif

//=================================================
//Test defines (tbd: move in osdTest.h ?)

//CDMA test chunk size
#ifndef PIECE
#define PIECE (8*1024)
#endif

#define TEST_DONE(x) (1000+x)
#define PING_TEST 1
#define CDMA_TEST 2
#define POLY_TEST 3

#endif //__OSD_DEFS_H__

