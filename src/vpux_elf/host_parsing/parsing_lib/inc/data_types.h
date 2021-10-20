#pragma once

#include <string>
#include <array>
#include <vector>
#include <stdint.h>
#include <stdio.h>

namespace parsing_lib {

// Class that encapsulates a type which may be present or not (think valid ptr or nullptr)
// Automatically updates validity through assignment and construction
template <typename T>
class Optional {
public:
    Optional<T>()
        : valid_(false) {
    }
    Optional<T>(const T &t)
        : valid_(true)
        , instance(t) {
    }
    void operator=(const T &other) {
        instance = other;
        valid_ = true;
    }
    const T *operator->() {
        return valid() ? &instance : nullptr;
    };
    bool operator()() { return valid_; }
    bool valid() { return valid_; }

private:
    bool valid_;
    T instance;
};

// The following are mostly copied from the flatbuffers schema, with some unused fields removed
// since we aren't aiming for binary compatibility
enum class MemoryLocation : uint8_t {
    NOT_SET = 0,
    ProgrammableInput = 1,
    ProgrammableOutput = 2,
    VPU_DDR_Heap = 3,
    GraphFile = 4,
    VPU_CMX_NN = 5,
    VPU_CMX_UPA = 6,
    VPU_DDR_BSS = 7,
    VPU_CSRAM = 8,
    AbsoluteAddr = 9,
    MAC_Accumulators = 10,
    ProfilingOutput = 11,
    GFEmbeddedKernel = 12,
    KernelsBuffer = 13
};

enum class DType : uint8_t {
    NOT_SET = 0,
    FP64 = 1,
    FP32 = 2,
    FP16 = 3,
    FP8 = 4,
    U64 = 5,
    U32 = 6,
    U16 = 7,
    U8 = 8,
    I64 = 9,
    I32 = 10,
    I16 = 11,
    I8 = 12,
    I4 = 13,
    I2 = 14,
    I4X = 15,
    BIN = 16,
    LOG = 17,
    I2X = 18,
    BFP16 = 19,
    U4 = 20,
};

struct IndirectDataReference {
    uint64_t data_index = 999999999999999999ULL;
    uint64_t sparsity_index = 999999999999999999ULL;
    uint64_t storage_element_index = 999999999999999999ULL;
    uint32_t storage_element_size = 0;
};

struct TensorReference {
    std::string name;
    std::vector<uint32_t> dimensions;
    std::vector<float> strides;
    uint32_t leading_offset;
    uint32_t trailing_offset;
    IndirectDataReference data;
    MemoryLocation locale;
    std::vector<uint32_t> locale_index;
    DType data_dtype;
    std::vector<uint8_t> quant_zero;
    std::vector<uint16_t> quant_mult;
    std::vector<uint8_t> quant_shift;
    uint8_t quant_post_shift_right = 0;
    uint64_t order = 0;
    uint8_t swizzling_key = 0;
    // TODO will need a different way of representing these if they're needed
    // Optional<TensorReference> dimsTensor;
    // Optional<TensorReference> stridesTensor;
    std::vector<uint16_t> base_ptrs;
};

struct BarrierReference {
    std::vector<uint32_t> wait_barriers;
    std::vector<uint32_t> update_barriers;
    std::vector<uint32_t> virtual_wait_barriers;
    std::vector<uint32_t> virtual_update_barriers;
};

enum class NN2Optimization : uint8_t {
    NONE = 0,
    // A Convolution that had a spatial kernel (e.g. 3x3) converted to 1x1
    SQUASHED_CONVOLUTION = 1
};

enum class DPULayerType : uint8_t {
    CONV = 0,
    DWCONV = 1,
    MAXPOOL = 2,
    AVEPOOL = 3,
    FCL = 4,
    ELTWISE = 5,
    IDENTITY = 6,
    CMCONV = 7
};

enum class PPELayerType : uint8_t {
    STORE,
    LOAD,
    CLEAR,
    NOOP,
    HALT,
    ADD,
    SUB,
    MULT,
    LRELU,
    LRELUX,
    LPRELU,
    MAXIMUM,
    MINIMUM,
    CEIL,
    FLOOR,
    AND,
    OR,
    XOR,
    NOT,
    ABS,
    NEG,
    POW,
    EXP,
    SIGMOID,
    TANH,
    SQRT,
    RSQRT,
    FLEXARB
};

enum class MPE_Mode : uint8_t {
    VECTOR = 0,       // 1x16x16 (8bit)
    MATRIX = 1,       // 4x4x16 (8bit)
    VECTOR_FP16 = 2,  // 1x4x16 (16bit)
    CUBOID_16x16 = 3, // NTH = 4, NTW=4, NTK = 4  (16, 4)
    CUBOID_8x16 = 4,  // NTH = 2, NTW=4, NTK = 8 (8, 8)
    CUBOID_4x16 = 5,  // NTH = 1, NTW=4, NTK = 16 (4, 16)
    NOP = 6
};

enum class PPERoundingMode : uint8_t {
    RNE = 0,  // Round to nearest, ties to even (Available in VPU2.6)
    RNTZ = 1, // Round to nearest, ties toward zero (Available in VPU2.6)
    RNAZ = 2, // Round to nearest, ties away from zero (Available in VPU2.6)
    RUP = 3,  // Round up (Available in VPU2.0)
};

enum class Permutation : uint8_t {
    ZXY = 0,
    ZYX = 1,
    YZX = 2,
    YXZ = 3,
    XZY = 4,
    XYZ = 5,
};

struct PPEFixedFunction {
    std::vector<PPELayerType> Ops;
    int32_t Clamp_Low = -2147483648;
    int32_t Clamp_High = 2147483647;
    int32_t Lrelu_Mult = 1;
    uint32_t Lrelu_Shift = 0;
};

struct PPETask {
    Optional<TensorReference> scale_data;
    PPEFixedFunction fixed_function;
    PPERoundingMode rounding;
};

struct Invariant {
    DPULayerType dpu_task_type;
    PPETask ppe_task;
    MPE_Mode mpe_frequent_mode;
    uint16_t kernelH;
    uint16_t kernelW;
    uint16_t kernel_strideH;
    uint16_t kernel_strideW;
    uint16_t kernel_padLeft;
    uint16_t kernel_padRight;
    uint16_t kernel_padTop;
    uint16_t kernel_padBottom;
    Optional<TensorReference> parent_input_tensor;
    Optional<TensorReference> parent_output_tensor;
    Optional<TensorReference> parent_weights_tensor;
    Optional<TensorReference> input_data;
    Optional<TensorReference> output_data;
    Optional<TensorReference> weights_data;
    Optional<TensorReference> weights_table;
    Optional<TensorReference> activation_window;
    int32_t activation_window_channel_length;
    // enabled_optimizations: [NN2Optimization];
    int32_t odu_offset = 0;
    int32_t out_channel_offset = 0;
    bool is_segmented = false;
    bool is_continued = false;
    bool is_superdense = false;
    // std::vector<uint16_t> segment_height;
};

struct Variant {
    BarrierReference associated_barriers;
    MPE_Mode mpe_mode;
    uint16_t padLeft;
    uint16_t padRight;
    uint16_t padTop;
    uint16_t padBottom;
    uint16_t workload_start_X;
    uint16_t workload_start_Y;
    uint16_t workload_start_Z;
    uint16_t workload_end_X;
    uint16_t workload_end_Y;
    uint16_t workload_end_Z;
    Optional<TensorReference> profiling_data;
};

struct NCE2Task {
    std::vector<Invariant> invariant;
    std::vector<Variant> variant;
};

struct Barrier {
    uint16_t barrier_id;
    uint16_t consumer_count;
    uint16_t producer_count;
};

struct DMATask {
  TensorReference src;
  TensorReference dst;
  bool compression = false;
  bool set_ord = false;
  bool set_crit = false;
};

}
