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
/// @file      NN_Types.h
/// 

#ifndef __NN_TYPES_H__
#define __NN_TYPES_H__

#include <stdint.h>

// TODO rename file - KMBFlicMsgInterface

struct flicTensorDescriptor_t {
    uint32_t n;
    uint32_t c;
    uint32_t w;
    uint32_t h;
    uint32_t totalSize;
    uint32_t widthStride;
    uint32_t heightStride;
    uint32_t channelsStride;
    uint32_t dtype; //to retrieve I/O precision
};

typedef struct BlobHandle_t {
    int32_t graphid;
    // const void *graphBuff;
    uint32_t graphBuff; // Changed to 32-bit p-address
    uint32_t graphLen;
    uint32_t refCount;
} BlobHandle_t;

typedef enum {
    U8 = 0, //unsigned int 8
    FP16 = 1, //floating point 16
    FP32 = 2, //floating point 32
    NOTKNOWN = 3,
}precision_t;

typedef enum {
    NHWC = 0, // row major - channel minor, for RGB image: RGB00, RGB01, RGB02,...
             // all RGB pixels by row
    NCHW = 1, // channel major - column minor (planar), for RGB image:
             // R01R02R03...G01G02G03...B01B02B03...
             // all Red rows..all Green rows..all Blue rows
    NHCW = 2, // row major - column minor (interleaved), for RGB image:
             // R00R01..R0k.., G00G01..G0k.., B00B01..B0k.., R10R11..R1k..
             // 1st Red row, 1st Green row, 1st Blue Rrw, 2nd Red row..
    NCWH = 3, // channel major - row minor, for RGB image:
             // R00R10R20... G00G10G20...B00B10B20...
             // all Red columns, all Green columns, all blue columns
    NWCH = 4, // column major - row minor; for RGB image:
             // R00R10..Rk0.., G00G10..Gk0.., B00B10..Bk0.., R01R11..Rk1..
             // 1st Red col, 1st Green col, 1st blue col, 2nd Red col...
    NWHC = 5, // column major - channle minor, for RGB image: RGB00, RGB10, RGB20...
             // all RGB pixels by col...
} Layout_t;

// struct TensorMsg:public PoBuf {
//   public:
//     uint8_t * data;
//     flicTensorDescriptor_t desc;
//     uint32_t streamId;
//     uint8_t InferenceNo;

//      TensorMsg() {
//         streamId = INVALID_STREAM_ID;
//         data = NULL;
//         size = 0;
//         InferenceNo = 0;
//       }
// };

// typedef PoPtr < TensorMsg > TensorMsgPtr;

// TODO This is from mvnci.h
typedef enum
{
    MVNCI_SUCCESS,
    MVNCI_WRONG_INPUT_FORMAT,
    MVNCI_UNSUPPORTED_NETWORK_ELEMENT,
    MVNCI_INVALID_HANDLE,
    MVNCI_OUT_OF_RESOURCES,
    MVNCI_NOT_IMPLEMENTED,
    MVNCI_INTERNAL_ERROR,
} MvNCIErrorCode;

#endif // __NN_TYPES_H__
