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

struct MvNCINetwork;
typedef struct MvNCINetwork *MvNCINetworkHandle;

struct MvNCIExecutor;
typedef struct MvNCIExecutor *MvNCIExecutorHandle;

typedef struct
{
    unsigned int tensors;
    unsigned int channels;
    unsigned int height;
    unsigned int width;
    unsigned int bitsPerPixel;
    unsigned int widthStride;
    unsigned int heightStride;
    unsigned int channelsStride;
    unsigned int tensorStride;
} MvNCITensorShape;

typedef struct
{
    unsigned char upaShaves;
} MvNCIProcessingResources;

typedef struct
{
    void *address;
    unsigned int size;
} MvNCIBuffer;

typedef struct
{
    MvNCIBuffer scratchBuffer;
    MvNCIBuffer prefetchBuffer;
} MvNCIMemoryResources;

typedef struct
{
    unsigned int cycles;
    unsigned int instructions;
    unsigned int stalls;
    unsigned int branches;
} MvNCIShaveCounters;

typedef struct
{
    unsigned int total_im_clock_cycles_;
    unsigned int total_ir_clock_cycles_;

    unsigned int numSoftLayerPerf_;
    unsigned int *irTicksPerSoftLayer_;
    MvNCIShaveCounters *softLayerPerf_;

    unsigned int barrier_count_;
    const unsigned int *barrier_lift_time_;
    const unsigned int *barrier_free_time_;
    const unsigned short *original_barrier_;
} MvNCIPerformanceData;

typedef enum
{
    /// No performance data, used to disable collection
    MVNCI_PERF_NONE            = 0,

    /// Full inference duration on Inference Manager level
    MVNCI_PERF_INFERENCE_OUTER = 1,

    /// Full inference duration on Inference Runtime level
    MVNCI_PERF_INFERENCE_INNER = 1 << 1,

    /// SW layer duration on Inference Runtime level
    MVNCI_PERF_SW_LAYER_OUTER  = 1 << 2,

    /// SW layer cycle counts
    MVNCI_PERF_SW_LAYER_INNER  = 1 << 3,

    /// HW layer duration on Inference Runtime level
    /// Reserved for later use
    /// MVNCI_PERF_HW_LAYER_OUTER  = 1 << 4,

    /// HW layer cycle counts
    /// Reserved for later use
    /// MVNCI_PERF_HW_LAYER_INNER  = 1 << 5,

    /// Barrier production time from Inference Runtime's ISR
    MVNCI_PERF_BARRIER_LIFT    = 1 << 6,

    /// Barrier consumption time from Inference Runtime's ISR
    MVNCI_PERF_BARRIER_FREE    = 1 << 7,

    /// Shorthand for collecting all data
    MVNCI_PERF_ALL             = ~0
} MvNCIPerformanceDataType;

typedef struct
{
    unsigned char major;
    unsigned char minor;
    unsigned char patch;
} MvNCIVersion;

#include <vector>

// physical addrs used in these structs for tensors:
struct NnExecMsg {
  public:
    unsigned int inferenceID;
    std::vector<uint32_t> inputTensors;
    std::vector<uint32_t> outputTensors;
};
typedef PoPtr<NnExecMsg> NnExecMsgPtr;

struct NnExecResponseMsg {
  public:
    unsigned int inferenceID;
    std::vector<uint32_t> outputTensors;
    MvNCIErrorCode status { MVNCI_SUCCESS };
};
typedef PoPtr<NnExecResponseMsg> NnExecResponseMsgPtr;


#endif // __NN_TYPES_H__
