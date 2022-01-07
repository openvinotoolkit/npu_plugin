/*
 * {% copyright %}
 */
#pragma once

#include <stdint.h>

namespace host_parsing {


// Act Kernel stuff
typedef uint32_t actKernelDataBuffer;
typedef uint32_t actKernelTextBuffer;
typedef uint32_t act_kernel_args;
typedef uint32_t actKernelEntry;
typedef uint32_t actRuntimeEntry;

enum ActWLType : uint8_t { WL_KERNEL = 0x00, WL_DEBUG = 0x04, WL_UNKNOWN };

// kernel params
typedef struct {
    uint16_t iw; // input width
    uint16_t ih; // input height
    uint16_t ic; // input channels
    uint16_t ow;
    uint16_t oh;
    uint16_t oc;
    uint16_t fw : 8;
    uint16_t fh : 8;
    uint16_t stride_w : 8;
    uint16_t stride_h : 8;
    uint32_t leaky_relu_alpha; // uint32, was float
}kernel_op; 

// pointers -> uint32_t
// embed kernel_op struct
typedef struct {
    kernel_op p_operation;
    uint32_t p_act_data;
    uint32_t p_act_out;
}slf_kernel_params;

typedef struct {
    slf_kernel_params params;
    kernel_op operations;
}slf_kernel_asds;


// Bit field for fine-grained configuration of DMA job
struct DmaConfigBits{
    uint64_t type : 2;              // Job type(1D/2D)
    uint64_t burst_length : 8;      // Burst length
    uint64_t critical : 1;          // Critical task
    uint64_t interrupt_en : 1;      // Interrupt enable
    uint64_t interrupt_trigger : 7; // Interrupt status id when task is executed
    uint64_t skip_nr : 7;           // Skip descriptor
    uint64_t order_forced : 1; // Force ordering. Dispatch the current task only after the previous task has completed
    uint64_t watermark_en : 1; // Job watermark enable
    uint64_t dec_en : 1;       // Decoder enable
    uint64_t barrier_en : 1;   // Barrier use enable
    uint64_t reserved : 34;    // Reserved
};

// Barrier configuration struct
struct DmaBarrierCfg {
    uint64_t prod_mask; // 64-bit mask depicting which barriers are affected by
                        // task completion
    uint64_t cons_mask; // 64-bit mask depicting which barriers are gating the current
                        // Link Agent
};

// 2D descriptor Attributes struct
struct Dma2DAttributes {
    uint32_t src_width; // Bytes of data required from one line of source
    int32_t src_stride; // Length in bytes from start of one line of data,
                        // to start of next line of data
    uint32_t dst_width; // Bytes of data required from one line of destination
    int32_t dst_stride; // Length in bytes from start of one line of data,
                        // to start of next line of data
};

// Generic descriptor type struct
struct alignas(64) DmaDescriptor {
    uint64_t link_address : 40; // pointer to the next element in linked list
    uint64_t reserved : 23;
    uint64_t watermark : 1; // watermark to indicate that the job has completed
    union {
        DmaConfigBits cfg_bits;
        uint64_t full_cfg_register;
    } cfg_link;
    uint64_t src;             // Address of the data transfer source
    uint64_t dst;             // Address of the data transfer destination
    uint32_t length;          // Job length
    uint32_t num_planes : 8;  // Number of planes
    uint32_t task_id : 24;    // Task id for the current job
    int32_t src_plane_stride; // Source plane stride
    int32_t dst_plane_stride; // Destination plane stride
    union {
        Dma2DAttributes attr2d;   // Attributes that apply for 2D jobs (i.e. striding)
        DmaBarrierCfg barriers1d; // Barrier mask configurations for 1D jobs
    };
    DmaBarrierCfg barriers; // Barrier mask configurations for 2D jobs
    // The descriptor must be aligned to 64 byte boundary
    // This is needed for L2 cache line alignment
};

#pragma pack(push, 1)

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
            unsigned int pool_wt_rd_dis : 1;
            unsigned int unused : 11;
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
} DPUInvariantRegisters;

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

} DPUVariantRegisters;

#pragma pack(pop)

/**
** HW related utility functions and enum values (enums correspond to HW register settings)
**/

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
    UNKNOWN
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
    UNKNOWN
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
    BF16 = FP16,
    UNKNOWN = 0x0A
};

enum class IDUNthw_Ntk : uint8_t {
    IDU_8_8 = 0,
    IDU_4_16 = 1,
    IDU_UNUSED = 2,
    IDU_16_4 = 3,
};

enum class ODUGrid : uint8_t { GRID_4x4, GRID_16x1, GRID_4x1 };

enum class ODUNthw : uint8_t {
    NTHW_1,
    NTHW_4,
    NTHW_8,
    NTHW_16,
};

enum MPEGrid { GRID_4x4, GRID_16x1 };
}
