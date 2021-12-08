/*
* {% copyright %}
*/
#ifndef _NN_TENSOR_ND_TYPES_H_
#define _NN_TENSOR_ND_TYPES_H_

enum {
    MAX_ND_DIMS = 15,
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
    NN_U4,
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

#endif  // _NN_TENSOR_ND_TYPES_H_
