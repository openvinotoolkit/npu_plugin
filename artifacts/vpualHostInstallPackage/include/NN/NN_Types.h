// {% copyright %}
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

// Maximal number of dimensions that TensorRefNDData can hold
constexpr int MAX_ND_DIMS = 15;

// All possible precisions that TensorRefNDData can hold
enum class DataType : uint32_t {
    NN_FP64,
    NN_FP32,
    NN_FP16,
    NN_FP8,
    NN_U64,
    NN_U32,
    NN_U16,
    NN_U8,
    NN_I64,
    NN_I32,
    NN_INT32=NN_I32,
    NN_I16,
    NN_INT16=NN_I16,
    NN_I8,
    NN_I4,
    NN_I2,
    NN_BIN,
    NN_UNDEFINED,
};

// All possible layouts that TensorRefNDData can hold
enum class NDOrder : uint64_t
{
    ND_NHWC = 0x1342,
    ND_NHCW = 0x1324,
    ND_NCHW = 0x1234,
    ND_NCWH = 0x1243,
    ND_NWHC = 0x1432,
    ND_NWCH = 0x1423,
    ND_HWC  = 0x231,
    ND_CHW  = 0x123,
    ND_WHC  = 0x321,
    ND_HCW  = 0x213,
    ND_WCH  = 0x312,
    ND_CWH  = 0x132,
    ND_NC   = 0x12,
    ND_CN   = 0x21,
    ND_C    = 0x1,
    ND_H    = 0x1,
    ND_W    = 0x1,

    FULL_ND_ORDER = 0x123456789ABCDEF,
    FULL_ND_NHWC = 0x123456789ABCEFD,
    ND_UNDEFINED = 0xffffffffffffffff,
};

// Structure to describe N-dimensional tensor reference data.
// TensorRefNDData is too large to be fit inside single VpualMessage.
// Thus, it is composed out of TensorDescDims and TensorDescStrides.
struct TensorRefNDData
{
    DataType dType = DataType::NN_FP16; // tensor precision
    NDOrder ndOrder = NDOrder::ND_NHWC; // tensor layout
    int32_t ndims = -1;                 // number of dimensions
    int32_t dims[MAX_ND_DIMS];          // tensor sizes in bytes
    int64_t stridesBits[MAX_ND_DIMS];   // tensor strides in bits
};

// Part of TensorRefNDData which describes tensor dimensions
typedef struct __attribute__((packed)) {
    int32_t ndims = -1;                 // number of dimensions (-1 means failed transmission)
    DataType dType = DataType::NN_FP16; // tensor precision
    int32_t dims[MAX_ND_DIMS];          // tensor sizes in bytes
} TensorDescDims;

// Part of TensorRefNDData which describes tensor strides
typedef struct __attribute__((packed)) {
    NDOrder ndOrder = NDOrder::ND_UNDEFINED;    // tensor layout (-1 means failed transmission)
    int64_t stridesBits[MAX_ND_DIMS];           // tensor strides in bits
} TensorDescStrides;

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
