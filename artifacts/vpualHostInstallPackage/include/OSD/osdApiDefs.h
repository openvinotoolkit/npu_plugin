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
/// @file      osdApiDefs.h
/// 

#ifndef __OSD_API_DEFS_H__
#define __OSD_API_DEFS_H__

//Leon/ARM api to build the task-list
#include <stdint.h>

//The only supported format really...
#define  F_NV12 0

//[Address type]
typedef unsigned char* OsdAddr;

typedef struct {
    OsdAddr  base; //list base addr
    OsdAddr  curr; //current addr (updates during parse process)
    uint32_t size; //list size (need to check when addin primitives)
    uint32_t num;  //number of primitives
}OsdList;

typedef struct {
    float x;
    float y;
}OsdPoint;

#define IDX_Y 0
#define IDX_C 1
#define IDX_A 2

typedef struct {
    uint32_t width;   //Y width
    uint32_t height;  //Y height
    uint16_t stride;  //Y ln stride
    uint16_t fmt;     //Chroma format (default 8bit NV12)
    OsdAddr  addr[3]; //Luma/Chroma/Alpha (see above IDX_Y,C,A)
}OsdBuff;

//##################################################################
//Main config structures
typedef struct {
    OsdPoint pos; //drawing position
    OsdBuff  bmp; //yuv+alpha
}OsdBmpDesc;

typedef struct {
    OsdPoint pos;    //drawing position
    OsdBuff  mask;   //alpha only
    uint32_t frColor;//AYUV
    uint32_t bgColor;//AYUV
    uint32_t charNfo;//for color invert {res:8 delta:8 width:8 height:8}
}OsdMaskDesc;

typedef struct { //[Host view]
    OsdPoint  pos;    //drawing position
    uint32_t  color;  //AYUV color
    uint32_t  nVerts; //number of vertices
    OsdPoint  *verts; //the vertex array
}OsdPolyDesc;

typedef struct { //[Shave view]
    OsdPoint  pos;
    uint32_t  color;
    uint32_t  nVerts;
    OsdPoint   verts;//1st element, this is an array actually
}OsdPolyDesc2;

typedef struct {
    OsdPoint pos;     //drawing position
    uint32_t width;   //must be multiple of cellSz
    uint32_t height;  //must be multiple of cellSz
    uint32_t cellSz;  //cell size (same for X, Y)
    uint32_t decim;   //0: no decimation
}OsdMosDesc;

typedef struct {
    OsdPoint pos;   //drawing position
    uint32_t width;
    uint32_t height;
    uint32_t thick;
    uint32_t color; //AYUV
}OsdFrmDesc;

typedef struct {
    OsdPoint pos;
    uint32_t width;
    uint32_t height;
    uint32_t color;
}OsdBoxDesc;

//##################################################################
//Flags
#define OFLG_EN_PIPE (1<<0) //Enable inter-primitive pipe
#define OFLG_VERBOSE (1<<1) //tbd
#define OFLG_xxxxxxx (1<<2) //tbd

//Flag values
#define OFLG_ON  1  //enable
#define OFLG_OFF 0  //disable

#endif