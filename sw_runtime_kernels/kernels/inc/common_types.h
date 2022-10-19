//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0

#ifndef COMMON_TYPES_H_
#define COMMON_TYPES_H_

#ifdef __cplusplus
namespace sw_params {
#endif

enum {
    MAX_ND_DIMS = 15,
    MAX_KERNEL_INPUTS = 8,
    MAX_KERNEL_OUTPUTS = 8,
};

enum DataType : uint32_t {
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
    NN_BF16,
    NN_UNDEFINED,
};

typedef uint64_t NDOrder;

typedef enum : uint64_t
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
    FULL_ND_NHWC = 0x123456789ABCEFD
} NDFrequentlyUsedOrders;

enum Location : uint32_t
{
    NONE,
    DDR,
    NN_CMX,
    UPA_CMX
};

#ifdef __cplusplus
    #define ALIGN_AS(size) alignas(size)
#else
    #define ALIGN_AS(size) __attribute__((aligned(size)))
#endif

#pragma pack(push, 1)
struct MemRefData {
    uint32_t dataAddr;      // Can't use pointers, since they have platform-dependent size.
                            // Will be located in WIN_F.

    uint32_t isStatic;      // Boolean flag to indicate static shape vs dynamic shape.

    uint32_t numDims;
    uint32_t dimsAddr;      // Pointer to the buffer with dimensions (int32_t[]).
    uint32_t stridesAddr;   // Pointer to the buffer with strides in bits (int64_t[]).
                            // Will be located in WIN_E (static case) or in WIN_F (dynamic case).
                            // The kernel should infer output dims/strides and write them only in dynamic case.

    uint32_t dataType;      // An enum, which should be aligned between kernels and the compiler.
    uint64_t dimsOrder;     // Packed permutation array.
    enum Location location;
};

struct ALIGN_AS(64) BaseKernelParams  {
    int32_t inputsOffset;
    uint32_t numInputs;
    int32_t outputsOffset;
    uint32_t numOutputs;
};

#pragma pack(pop)
#undef ALIGN_AS

#ifdef __cplusplus
}  // namespace sw_params
#endif

#endif  // COMMON_TYPES_H_
