/*
* {% copyright %}
*/
#ifndef NN_RUNTIME_TYPES_H_3600__
#define NN_RUNTIME_TYPES_H_3600__

#include <stdint.h>
#include <assert.h>
#ifdef CONFIG_TARGET_SOC_3720
#include <nn_relocation_3720.h>
#include <nn_hw_resources.h>
#include <nn_runtime_configs.h>
#else
#include <nn_resources.h>
#endif
#include <nn_log.h>

namespace nn {
namespace common_runtime {
// NCE_DPU_TENSOR_MODE wmode&amode - IDU types for activation&weights
enum class InputTensorDType : uint8_t {
    FP16 = 0x0,
    U8 = 0x1,
    I8 = 0x2,
    I4 = 0x3,
    I2 = 0x4,
    BF16 = 0x5,
    U4 = 0x6,
    BIN = 0x7,
    FP8 = 0x8,
    INPUT_DTYPE_UNKNOWN
};

// NCE_DPU_MPE_CFG mpe_mode
enum class MpeActivationWeightDtype : uint8_t {
    FP16 = 0x0,
    U8 = 0x1,
    I8 = 0x2,
    I4 = 0x3,
    I2 = 0x4,
    I4X = 0x5, // not used
    I2X = 0x6, // not used
    BIN = 0x7,
    MPE_DTYPE_UNKNOWN
};

// NCE_DPU_ODU_CFG  dtype 4 bit field
enum class OutputTensorDType : uint8_t {
    FP16 = 0x00,
    U8F = 0x01,
    G8 = 0x02,
    I8 = 0x03,
    I32 = 0x04,
    I4 = 0x05,
    I2 = 0x06,
    LOG = 0x07,
    BIN = 0x08,
    FP32 = I32, // The FP32 is same as I32 since NCE_DPU_ODU_CFG enums are based on dtype size
    U4 = I4, // The U4 is same as I4 since NCE_DPU_ODU_CFG enums are based on dtype size
    OUTPUT_DTYPE_UNKNOWN = 0x0A
};

inline uint8_t getIDUDTypeSizeBits(InputTensorDType dtype) {
    switch (dtype) {
        case InputTensorDType::FP16:
            return 16;
        case InputTensorDType::U8:
            return 8;
        case InputTensorDType::I8:
            return 8;
        case InputTensorDType::I4:
            return 4;
        case InputTensorDType::I2:
            return 2;
        case InputTensorDType::BF16:
            return 16;
        case InputTensorDType::U4:
            return 4;
        case InputTensorDType::FP8:
            return 8;
        case InputTensorDType::BIN:
            return 1;
        default:
            nnLog(MVLOG_ERROR, "Unknown IDU data type");
            return 1;
    }
}

typedef uint64_t PhysicalBarrierMask;

struct BarrierUserConfig {
    PhysicalBarrierMask wait_mask_;
    PhysicalBarrierMask post_mask_;
    unsigned short start_after_;
    unsigned short clean_after_;
    unsigned int virtual_dep_;
};

struct BarrierGpioConfig {
    unsigned char group_;
    unsigned char mask_;
};
} // namespace common_runtime

namespace dpu_runtime {
enum {
    PPE_ILIST_ENTRIES = 16,
};

typedef struct {
    /* IDU */
    struct {
        unsigned int se_addr;
        unsigned int sparsity_addr;
    } se_sp_addr[4];

    union {
        unsigned int se_sp_size;
        struct {
            unsigned int sp_seg_size : 14;
            unsigned int se_seg_size : 18;
        } se_sp_size_bf;
    } se_sp_size[3];

    union // new field
    {
        unsigned int z_config;
        struct {
            unsigned int se_z_split : 4;
            unsigned int num_ses_in_z_dir : 9;
            unsigned int cm_sp_pattern : 16;
            unsigned int reserved : 2;
            unsigned int addr_format_sel : 1;
        } z_config_bf;
    } z_config;

    union {
        unsigned int kernel_pad_cfg;
        struct {
            unsigned int mpe_assign : 1;
            unsigned int pad_right_en : 1;
            unsigned int pad_left_en : 1;
            unsigned int pad_bottom_en : 1;
            unsigned int pad_top_en : 1;
            unsigned int kernel_y : 4;
            unsigned int kernel_x : 4;
            unsigned int wt_plt_cfg : 2;
            unsigned int act_dense : 1;
            unsigned int wt_dense : 1;
            unsigned int stride_y_en : 1;
            unsigned int stride_y : 3;
            unsigned int dynamic_bw_en : 1;
            unsigned int dw_wt_sp_ins : 1;
            unsigned int layer1_wt_sp_ins : 1;
            unsigned int layer1_cmp_en : 1;
            unsigned int pool_opt_en : 1;
            unsigned int unused1 : 3;
            unsigned int sp_se_tbl_segment : 1;
            unsigned int rst_ctxt : 1;
            unsigned int unused2 : 1;
        } kernel_pad_cfg_bf;
    } kernel_pad_cfg;

    // Placeholders: both weight_size and weight_num are used to pad this
    // struct up to the full size of the invariant registers in the
    // NCE DPU registers.
    // Both registers are variant for each workload, so they are only to
    // be used from the DPUVariantRegisters struct.
    unsigned int weight_size_placeholder;
    unsigned int weight_num_placeholder;

    unsigned int weight_start;

    union {
        unsigned int tensor_size0;
        struct {
            unsigned int tensor_size_x : 14;
            unsigned int tensor_size_y : 14;
            unsigned int unused : 4;
        } tensor_size0_bf;
    } tensor_size0;

    union {
        unsigned int tensor_size1;
        struct {
            unsigned int tensor_size_z : 14;
            unsigned int unused : 18;
        } tensor_size1_bf;
    } tensor_size1;

    unsigned int tensor_start;

    union // removed field addr_format_sel
    {
        unsigned int tensor_mode;
        struct {
            unsigned int wmode : 4;
            unsigned int amode : 4;
            unsigned int stride : 3;
            unsigned int zm_input : 1;
            unsigned int dw_input : 1;
            unsigned int cm_input : 1;
            unsigned int workload_operation : 2;
            unsigned int pad_value : 16;
        } tensor_mode_bf;
    } tensor_mode;

    unsigned int elop_sparsity_addr;
    unsigned int elop_se_addr;

    union // new field elop_wload_type
    {
        unsigned int elops_wload;
        struct {
            unsigned int elop_wload : 1;
            unsigned int seed_wload : 1;
            unsigned int fifo_wr_wload : 1;
            unsigned int elop_wload_type : 1;
            unsigned int pool_wt_data : 16;
            unsigned int unused1 : 6;
            unsigned int pool_wt_rd_dis : 1;
            unsigned int unused2 : 5;
        } elops_wload_bf;
    } elops_wload;

    unsigned int act_offset[4];
    unsigned int base_offset_a;
    unsigned int base_offset_b;
    unsigned int wt_offset;

    /* ODU*/
    union // new fields
    {
        unsigned int odu_cfg;
        struct {
            unsigned int dtype : 4;
            unsigned int reserved_0 : 1;
            unsigned int sp_value : 8;
            unsigned int sp_out_en : 1;
            unsigned int reserved_1 : 1;
            unsigned int write_sp : 1;
            unsigned int write_pt : 1;
            unsigned int write_ac : 1;
            unsigned int mode : 2;
            unsigned int grid : 1;
            unsigned int swizzle_key : 3;
            unsigned int reserved_2 : 1;
            unsigned int nthw : 2;
            unsigned int permutation : 3;
            unsigned int debug_mode : 1;
            unsigned int reserved_3 : 1;
        } odu_cfg_bf;
    } odu_cfg;

    unsigned int odu_be_size; // fromerly ODU_CTX_SIZE
    unsigned int odu_be_cnt;  // formerly OCU_CTX_THRESHOLD
    unsigned int se_size;

    union // renamed fields
    {
        unsigned int te_dim0;
        struct {
            unsigned int te_dim_y : 13;
            unsigned int te_dim_z : 13;
            unsigned int unused : 6;
        } te_dim0_bf;
    } te_dim0;
    union // renamed fields
    {
        unsigned int te_dim1;
        struct {
            unsigned int te_dim_x : 13;
            unsigned int unused : 19;
        } te_dim1_bf;
    } te_dim1;

    unsigned int pt_base;
    unsigned int sp_base;
    unsigned int base_ptr_a;
    unsigned int base_ptr_b;
    unsigned int base_adr[4];

    union {
        unsigned int odu_cast;
        struct {
            unsigned int cast_enable : 1;
            unsigned int reserved : 3;
            unsigned int cast_offset : 28;
        } odu_cast_bf;
    } odu_cast[3];

    /* MPE */
    union // new fields
    {
        unsigned int mpe_cfg;
        struct {
            unsigned int mpe_wtbias : 8;
            unsigned int mpe_actbias : 8;
            unsigned int mpe_mode : 3;
            unsigned int mpe_dense : 1;
            unsigned int mrm_weight_dense : 1;
            unsigned int mrm_act_dense : 1;
            unsigned int mpe_daz : 1;
            unsigned int mpe_ftz : 1;
            unsigned int unused : 8;
        } mpe_cfg_bf;
    } mpe_cfg;

    unsigned int mpe_bus_data_sel;

    union // new struct union
    {
        unsigned int elop_scale;
        struct {
            unsigned int elop_scale_b : 16;
            unsigned int elop_scale_a : 16;
        } elop_scale_bf;
    } elop_scale;

    /* PPE */
    union // handled, new fields added
    {
        unsigned int ppe_cfg;
        struct {
            unsigned int ppe_g8_bias_c : 9;
            unsigned int ppe_g8_bias_b : 9;
            unsigned int ppe_g8_bias_a : 9;
            unsigned int unused : 5;
        } ppe_cfg_bf;
    } ppe_cfg;

    unsigned int ppe_bias; // no change
    union                  // new fields added
    {
        unsigned int ppe_scale;
        struct {
            unsigned int unused : 2;
            unsigned int ppe_scale_shift : 6;
            unsigned int unused1 : 2;
            unsigned int ppe_scale_round : 2;
            unsigned int unused2 : 4;
            unsigned int ppe_scale_mult : 16;
        } ppe_scale_bf;
    } ppe_scale;

    union // new field added
    {
        unsigned int ppe_scale_ctrl;
        struct {
            unsigned int ppe_scale_override : 1;
            unsigned int ppe_fp_scale_override : 1;
            unsigned int unused : 30;
        } ppe_scale_ctrl_bf;
    } ppe_scale_ctrl;

    union // new fields added
    {
        unsigned int ppe_prelu;
        struct {
            unsigned int unused : 8;          // 0-7
            unsigned int ppe_prelu_shift : 5; // 8-12
            unsigned int unused1 : 3;         // 13-15
            unsigned int ppe_prelu_mult : 11; // 16-26
            unsigned int unused2 : 5;         // 27-31
        } ppe_prelu_bf;
    } ppe_prelu;

    unsigned int vpu2p0_rsvd_1; // new
    unsigned int vpu2p0_rsvd_2; // new

    unsigned int ppe_scale_hclamp;
    unsigned int ppe_scale_lclamp;

    unsigned int vpu2p0_rsvd_3; // new

    union // added new fields
    {
        unsigned int ppe_misc;
        struct {
            unsigned int unused : 6;
            unsigned int ppe_fp16_ftz : 1;
            unsigned int ppe_fp16_clamp : 1;
            unsigned int ppe_i32_convert : 2;
            unsigned int unused1 : 22;
        } ppe_misc_bf;
    } ppe_misc;

    unsigned int ppe_fp_bias;  // new
    unsigned int ppe_fp_scale; // new
    unsigned int ppe_fp_prelu; // new

    union // new
    {
        unsigned int ppe_fp_cfg;
        struct {
            unsigned int ppe_fp_convert : 3;
            unsigned int ppe_fp_bypass : 1;
            unsigned int ppe_bf16_round : 1;
            unsigned int ppe_fp_prelu_en : 1;
            unsigned int unused : 26;
        } ppe_fp_cfg_bf;
    } ppe_fp_cfg;
} DPU_workload_base_register_config;

typedef struct {
    union {
        unsigned int workload_size0;
        struct {
            unsigned int workload_size_x : 14;
            unsigned int workload_size_y : 14;
            unsigned int unused : 4;
        } workload_size0_bf;
    } workload_size0;

    union {
        unsigned int workload_size1;
        struct {
            unsigned int workload_size_z : 14;
            unsigned int pad_count_up : 3;
            unsigned int pad_count_left : 3;
            unsigned int pad_count_down : 3;
            unsigned int pad_count_right : 3;
            unsigned int unused : 6;
        } workload_size1_bf;
    } workload_size1;

    union {
        unsigned int workload_start0;
        struct {
            unsigned int workload_start_x : 14;
            unsigned int workload_start_y : 14;
            unsigned int unused : 4;
        } workload_start0_bf;
    } workload_start0;

    union {
        unsigned int workload_start1;
        struct {
            unsigned int workload_start_z : 14;
            unsigned int unused : 18;
        } workload_start1_bf;
    } workload_start1;

    union {
        unsigned int offset_addr;
        struct {
            unsigned int nthw_ntk : 2;
            unsigned int bin_cfg : 1;
            unsigned int conv_cond : 1;
            unsigned int dense_se : 1;
            unsigned int idx_quad : 1;
            unsigned int swizzle_key : 3;
            unsigned int idu_mrm_clk_en : 1;
            unsigned int odu_clk_en : 1;
            unsigned int mpe_clk_en : 1;
            unsigned int ppe_clk_en : 1;
            unsigned int odu_stat_en : 1;
            unsigned int idu_stat_en : 1;
            unsigned int reserved_1 : 1;
            unsigned int odu_stat_clr_mode : 1;
            unsigned int idu_stat_clr_mode : 1;
            unsigned int reserved_2 : 1;
            unsigned int shave_l2_cache_en : 1;
            unsigned int idu_dbg_en : 2;
            unsigned int reserved_3 : 5;
            unsigned int wt_swizzle_key : 3;
            unsigned int wt_swizzle_sel : 1;
            unsigned int reserved_4 : 1;
        } offset_addr_bf;
    } offset_addr;

    union {
        unsigned int te_end0;
        struct {
            unsigned int te_end_y : 13;
            unsigned int te_end_z : 13;
            unsigned int unused : 6;
        } te_end0_bf;
    } te_end0;

    union {
        unsigned int te_end1;
        struct {
            unsigned int te_end_x : 13;
            unsigned int unused : 19;
        } te_end1_bf;
    } te_end1;

    union {
        unsigned int te_beg0;
        struct {
            unsigned int te_beg_y : 13;
            unsigned int te_beg_z : 13;
            unsigned int unused : 6;
        } te_beg0_bf;
    } te_beg0;

    union {
        unsigned int te_beg1;
        struct {
            unsigned int te_beg_x : 13;
            unsigned int unused : 19;
        } te_beg1_bf;
    } te_beg1;

    unsigned int weight_size;
    unsigned int weight_num;

} DPU_workitem_register_config;

typedef DPU_workload_base_register_config DPUInvariantRegisters;
typedef DPU_workitem_register_config DPUVariantRegisters;

enum IDUNthw_Ntk {
    IDU_NTHW_NTK_8_8 = 0,
    IDU_NTHW_NTK_4_16 = 1,
    IDU_NTHW_NTK_UNUSED = 2,
    IDU_NTHW_NTK_16_4 = 3,
};

enum DPULayerTypes {
    NO_OP,
    DPU_CONV,
    DPU_DWCONV,
    DPU_MAXPOOL,
    DPU_AVEPOOL,
    DPU_ELTWISE,
    DPU_CMCONV,
};

enum ODUGrid { ODU_GRID_4x4, ODU_GRID_16x1, ODU_GRID_4x1 };

enum ODUNthw {
    ODU_NTHW_1,
    ODU_NTHW_4,
    ODU_NTHW_8,
    ODU_NTHW_16,
};

enum MPEGrid { MPE_GRID_4x4, MPE_GRID_16x1 };

typedef common_runtime::BarrierUserConfig BarrierUserConfig;
typedef common_runtime::BarrierGpioConfig BarrierGpioConfig;

struct DPUInvariant {
    DPUInvariantRegisters registers_;
    BarrierUserConfig barriers_;
    BarrierGpioConfig barriers_gpio_;
    const unsigned int *ppe_list_;
    unsigned char ppe_list_batch_size_;
    unsigned char cluster_;
    unsigned short task_id_;
    unsigned int channel_major_stride_;
    unsigned int output_sparsity_offset_;
    bool isContConv_;
};

struct DPUAddresses {
    common_runtime::RelativeAddress input_;
    common_runtime::RelativeAddress output_;
    common_runtime::RelativeAddress weightsTable_; // map of used weights
    common_runtime::RelativeAddress weights_;      // Weights set - WT_OFFSET+DATA_PTR
    common_runtime::RelativeAddress ppe_list_;
};

struct DPUVariant {
    const DPUInvariant *invariant_;
    DPUVariantRegisters registers_;
    unsigned int output_sparsity_offset_;
    unsigned int weight_table_offset_;
    unsigned char cluster_;
    unsigned short task_id_;
};

union se_table_entry {
    uint32_t data;
    struct {
        uint32_t base_idx : 9;
        uint32_t offset : 20;
        uint32_t c2 : 1;
        uint32_t empty : 1;
        uint32_t c1 : 1;
    };
};
} // namespace dpu_runtime

namespace act_runtime {

typedef common_runtime::PhysicalBarrierMask BarrierMask;
typedef common_runtime::BarrierUserConfig BarrierUserConfig;
typedef common_runtime::BarrierGpioConfig BarrierGpioConfig;

typedef void *actKernelDataBuffer;
typedef void *actKernelTextBuffer;
typedef void act_kernel_args;
typedef void (*actKernelEntry)(act_kernel_args *);
typedef void (*actRuntimeEntry)(const uint32_t);

// these are going to be done via ctrl messages, but special tasks like loops may stay here
// refactor/remove in the futore
///@deprecated
enum class ActWLType : uint8_t { WL_KERNEL = 0x00, WL_DEBUG = 0x04, WL_UNKNOWN };

#ifdef NN_ENABLE_CONTEXT_DEBUGGING
enum class ActDebug : uint32_t {
    INVALID = 0x00,
    DEBUG_ACK = 0x1,
    DEBUG_FIFO_CONFIG_GLOBAL = 0x2,
    DEBUG_FIFO_WRITE = 0x3,
    DEBUG_FIFO_WRONG_READ = 0x4,
    DEBUG_FIFO_WRONG_WRITE = 0x5,
    DEBUG_VALID_BARRIER = 0x6,
    DEBUG_INVALID_BARRIER_P = 0x7,
    DEBUG_INVALID_BARRIER_C = 0x8,
    DEBUG_CACHE_FLUSH = 0x9,
    DEBUG_CACHE_INVALIDATE = 0xA,
    DEBUG_ACK_WAIT = 0xB,
    DEBUG_FIFO_CLEAR = 0xC,
};
#endif

extern "C" struct ActKernelRange {
    ActWLType type_{ActWLType::WL_UNKNOWN};
    actKernelEntry kernelEntry_{nullptr};
    actKernelTextBuffer textWindowBase_{nullptr};

    uint32_t codeSize_{0};
    uint32_t dataSecSize_{0};

#ifdef NN_ENABLE_CONTEXT_DEBUGGING
    ActDebug dbg_type_{ActDebug::INVALID};
    volatile uint32_t dbg0_{0};
    volatile uint32_t dbg1_{0};
    volatile uint32_t dbg2_{0};
    volatile uint32_t dbg3_{0};
#endif
};

extern "C" struct ActKernelInvocation {
    ActKernelRange *range_{nullptr};
    act_kernel_args *kernelArgs_{nullptr};
    actKernelDataBuffer dataWindowBase_{nullptr};

    BarrierUserConfig barriers_{};
    BarrierGpioConfig barriers_gpio_{};
    unsigned int invo_index_{0};
};
extern "C" struct ActKernelRuntimeConfigs {
    unsigned int stackFrames_[4/*common_runtime::AS_TOTAL*/]{0};
    unsigned int stackSize_{0};
    bool useScheduleEmbeddedRt_{false};

    // when useScheduleEmbeddedRt = true
    // this is a windowed address
    act_runtime::actRuntimeEntry runtimeEntry_{nullptr};

    // when useScheduleEmbeddedRt = false; FW copies ActRt to this buffer
    // when useScheduleEmbeddedRt = true; buffer already contains the ActRt
    unsigned char *actRtWindowBase_{nullptr};
    unsigned int codeWindowBufferSize_{0};
};
} // namespace act_runtime
namespace util {
class TaskContext;
}
} // namespace nn

#endif //NN_RUNTIME_TYPES_H_3600__
