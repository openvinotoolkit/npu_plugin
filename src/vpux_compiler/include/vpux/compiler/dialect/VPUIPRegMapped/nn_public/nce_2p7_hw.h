//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

//

#pragma once

#include <stdint.h>

namespace nn_public {

#pragma pack(push, 1)

typedef struct {
    // IDU
    struct {
        uint32_t se_addr;
        uint32_t sparsity_addr;
    } se_sp_addr[4];

    union {
        uint32_t se_sp_size;
        struct {
            uint32_t sp_seg_size : 14;
            uint32_t se_seg_size : 18;
        } se_sp_size_bf;
    } se_sp_size[3];

    union // new field
    {
        uint32_t z_config;
        struct {
            uint32_t se_z_split : 4;
            uint32_t num_ses_in_z_dir : 9;
            uint32_t cm_sp_pattern : 16;
            uint32_t reserved : 2;
            uint32_t addr_format_sel : 1;
        } z_config_bf;
    } z_config;

    union {
        uint32_t kernel_pad_cfg;
        struct {
            uint32_t mpe_assign : 1;
            uint32_t pad_right_en : 1;
            uint32_t pad_left_en : 1;
            uint32_t pad_bottom_en : 1;
            uint32_t pad_top_en : 1;
            uint32_t kernel_y : 4;
            uint32_t kernel_x : 4;
            uint32_t wt_plt_cfg : 2;
            uint32_t act_dense : 1;
            uint32_t wt_dense : 1;
            uint32_t stride_y_en : 1;
            uint32_t stride_y : 3;
            uint32_t dynamic_bw_en : 1;
            uint32_t dw_wt_sp_ins : 1;
            uint32_t layer1_wt_sp_ins : 1;
            uint32_t layer1_cmp_en : 1;
            uint32_t pool_opt_en : 1;
            uint32_t unused1 : 3;
            uint32_t sp_se_tbl_segment : 1;
            uint32_t rst_ctxt : 1;
            uint32_t unused2 : 1;
        } kernel_pad_cfg_bf;
    } kernel_pad_cfg;

    // Placeholders: both weight_size and weight_num are used to pad this
    // struct up to the full size of the invariant registers in the
    // NCE DPU registers.
    // Both registers are variant for each workload, so they are only to
    // be used from the DPUVariantRegisters struct.
    uint32_t weight_size_placeholder;
    uint32_t weight_num_placeholder;

    uint32_t weight_start;

    union {
        uint32_t tensor_size0;
        struct {
            uint32_t tensor_size_x : 14;
            uint32_t tensor_size_y : 14;
            uint32_t unused : 4;
        } tensor_size0_bf;
    } tensor_size0;

    union {
        uint32_t tensor_size1;
        struct {
            uint32_t tensor_size_z : 14;
            uint32_t unused : 18;
        } tensor_size1_bf;
    } tensor_size1;

    uint32_t tensor_start;

    union // removed field addr_format_sel
    {
        uint32_t tensor_mode;
        struct {
            uint32_t wmode : 4;
            uint32_t amode : 4;
            uint32_t stride : 3;
            uint32_t zm_input : 1;
            uint32_t dw_input : 1;
            uint32_t cm_input : 1;
            uint32_t workload_operation : 2;
            uint32_t pad_value : 16;
        } tensor_mode_bf;
    } tensor_mode;

    uint32_t elop_sparsity_addr;
    uint32_t elop_se_addr;

    union // new field elop_wload_type
    {
        uint32_t elops_wload;
        struct {
            uint32_t elop_wload : 1;
            uint32_t seed_wload : 1;
            uint32_t fifo_wr_wload : 1;
            uint32_t elop_wload_type : 1;
            uint32_t pool_wt_data : 16;
            uint32_t unused1 : 6;
            uint32_t pool_wt_rd_dis : 1;
            uint32_t unused2 : 5;
        } elops_wload_bf;
    } elops_wload;

    uint32_t act_offset[4];
    uint32_t base_offset_a;
    uint32_t base_offset_b;
    uint32_t wt_offset;

    // ODU
    union // new fields
    {
        uint32_t odu_cfg;
        struct {
            uint32_t dtype : 4;
            uint32_t reserved_0 : 1;
            uint32_t sp_value : 8;
            uint32_t sp_out_en : 1;
            uint32_t reserved_1 : 1;
            uint32_t write_sp : 1;
            uint32_t write_pt : 1;
            uint32_t write_ac : 1;
            uint32_t mode : 2;
            uint32_t grid : 1;
            uint32_t swizzle_key : 3;
            uint32_t reserved_2 : 1;
            uint32_t nthw : 2;
            uint32_t permutation : 3;
            uint32_t debug_mode : 1;
            uint32_t reserved_3 : 1;
        } odu_cfg_bf;
    } odu_cfg;

    uint32_t odu_be_size; // fromerly ODU_CTX_SIZE
    uint32_t odu_be_cnt;  // formerly OCU_CTX_THRESHOLD
    uint32_t se_size;

    union // renamed fields
    {
        uint32_t te_dim0;
        struct {
            uint32_t te_dim_y : 13;
            uint32_t te_dim_z : 13;
            uint32_t unused : 6;
        } te_dim0_bf;
    } te_dim0;
    union // renamed fields
    {
        uint32_t te_dim1;
        struct {
            uint32_t te_dim_x : 13;
            uint32_t unused : 19;
        } te_dim1_bf;
    } te_dim1;

    uint32_t pt_base;
    uint32_t sp_base;
    uint32_t base_ptr_a;
    uint32_t base_ptr_b;
    uint32_t base_adr[4];

    union {
        uint32_t odu_cast;
        struct {
            uint32_t cast_enable : 1;
            uint32_t reserved : 3;
            uint32_t cast_offset : 28;
        } odu_cast_bf;
    } odu_cast[3];

    // MPE
    union // new fields
    {
        uint32_t mpe_cfg;
        struct {
            uint32_t mpe_wtbias : 8;
            uint32_t mpe_actbias : 8;
            uint32_t mpe_mode : 3;
            uint32_t mpe_dense : 1;
            uint32_t mrm_weight_dense : 1;
            uint32_t mrm_act_dense : 1;
            uint32_t mpe_daz : 1;
            uint32_t mpe_ftz : 1;
            uint32_t unused : 8;
        } mpe_cfg_bf;
    } mpe_cfg;

    uint32_t mpe_bus_data_sel;

    union // new struct union
    {
        uint32_t elop_scale;
        struct {
            uint32_t elop_scale_b : 16;
            uint32_t elop_scale_a : 16;
        } elop_scale_bf;
    } elop_scale;

    // PPE
    union // handled, new fields added
    {
        uint32_t ppe_cfg;
        struct {
            uint32_t ppe_g8_bias_c : 9;
            uint32_t ppe_g8_bias_b : 9;
            uint32_t ppe_g8_bias_a : 9;
            uint32_t unused : 5;
        } ppe_cfg_bf;
    } ppe_cfg;

    uint32_t ppe_bias; // no change
    union              // new fields added
    {
        uint32_t ppe_scale;
        struct {
            uint32_t unused : 2;
            uint32_t ppe_scale_shift : 6;
            uint32_t unused1 : 2;
            uint32_t ppe_scale_round : 2;
            uint32_t unused2 : 4;
            uint32_t ppe_scale_mult : 16;
        } ppe_scale_bf;
    } ppe_scale;

    union // new field added
    {
        uint32_t ppe_scale_ctrl;
        struct {
            uint32_t ppe_scale_override : 1;
            uint32_t ppe_fp_scale_override : 1;
            uint32_t unused : 30;
        } ppe_scale_ctrl_bf;
    } ppe_scale_ctrl;

    union // new fields added
    {
        uint32_t ppe_prelu;
        struct {
            uint32_t unused : 8;          // 0-7
            uint32_t ppe_prelu_shift : 5; // 8-12
            uint32_t unused1 : 3;         // 13-15
            uint32_t ppe_prelu_mult : 11; // 16-26
            uint32_t unused2 : 5;         // 27-31
        } ppe_prelu_bf;
    } ppe_prelu;

    uint32_t vpu2p0_rsvd_1; // new
    uint32_t vpu2p0_rsvd_2; // new

    uint32_t ppe_scale_hclamp;
    uint32_t ppe_scale_lclamp;

    uint32_t vpu2p0_rsvd_3; // new

    union // added new fields
    {
        uint32_t ppe_misc;
        struct {
            uint32_t unused : 6;
            uint32_t ppe_fp16_ftz : 1;
            uint32_t ppe_fp16_clamp : 1;
            uint32_t ppe_i32_convert : 2;
            uint32_t unused1 : 22;
        } ppe_misc_bf;
    } ppe_misc;

    uint32_t ppe_fp_bias;  // new
    uint32_t ppe_fp_scale; // new
    uint32_t ppe_fp_prelu; // new

    union // new
    {
        uint32_t ppe_fp_cfg;
        struct {
            uint32_t ppe_fp_convert : 3;
            uint32_t ppe_fp_bypass : 1;
            uint32_t ppe_bf16_round : 1;
            uint32_t ppe_fp_prelu_en : 1;
            uint32_t unused : 26;
        } ppe_fp_cfg_bf;
    } ppe_fp_cfg;

} DPUInvariantRegisters;

typedef struct {
    union {
        uint32_t workload_size0;
        struct {
            uint32_t workload_size_x : 14;
            uint32_t workload_size_y : 14;
            uint32_t unused : 4;
        } workload_size0_bf;
    } workload_size0;

    union {
        uint32_t workload_size1;
        struct {
            uint32_t workload_size_z : 14;
            uint32_t pad_count_up : 3;
            uint32_t pad_count_left : 3;
            uint32_t pad_count_down : 3;
            uint32_t pad_count_right : 3;
            uint32_t unused : 6;
        } workload_size1_bf;
    } workload_size1;

    union {
        uint32_t workload_start0;
        struct {
            uint32_t workload_start_x : 14;
            uint32_t workload_start_y : 14;
            uint32_t unused : 4;
        } workload_start0_bf;
    } workload_start0;

    union {
        uint32_t workload_start1;
        struct {
            uint32_t workload_start_z : 14;
            uint32_t unused : 18;
        } workload_start1_bf;
    } workload_start1;

    union {
        uint32_t offset_addr;
        struct {
            uint32_t nthw_ntk : 2;
            uint32_t bin_cfg : 1;
            uint32_t conv_cond : 1;
            uint32_t dense_se : 1;
            uint32_t idx_quad : 1;
            uint32_t swizzle_key : 3;
            uint32_t idu_mrm_clk_en : 1;
            uint32_t odu_clk_en : 1;
            uint32_t mpe_clk_en : 1;
            uint32_t ppe_clk_en : 1;
            uint32_t odu_stat_en : 1;
            uint32_t idu_stat_en : 1;
            uint32_t reserved_1 : 1;
            uint32_t odu_stat_clr_mode : 1;
            uint32_t idu_stat_clr_mode : 1;
            uint32_t reserved_2 : 1;
            uint32_t shave_l2_cache_en : 1;
            uint32_t idu_dbg_en : 2;
            uint32_t reserved_3 : 5;
            uint32_t wt_swizzle_key : 3;
            uint32_t wt_swizzle_sel : 1;
            uint32_t reserved_4 : 1;
        } offset_addr_bf;
    } offset_addr;

    union {
        uint32_t te_end0;
        struct {
            uint32_t te_end_y : 13;
            uint32_t te_end_z : 13;
            uint32_t unused : 6;
        } te_end0_bf;
    } te_end0;

    union {
        uint32_t te_end1;
        struct {
            uint32_t te_end_x : 13;
            uint32_t unused : 19;
        } te_end1_bf;
    } te_end1;

    union {
        uint32_t te_beg0;
        struct {
            uint32_t te_beg_y : 13;
            uint32_t te_beg_z : 13;
            uint32_t unused : 6;
        } te_beg0_bf;
    } te_beg0;

    union {
        uint32_t te_beg1;
        struct {
            uint32_t te_beg_x : 13;
            uint32_t unused : 19;
        } te_beg1_bf;
    } te_beg1;

    uint32_t weight_size;
    uint32_t weight_num;

} DPUVariantRegisters;

#pragma pack(pop)

// HW related utility functions and enum values (enums correspond to HW register settings)

// base resources
constexpr uint32_t MAX_TILES = 2;
constexpr uint32_t DPU_PER_TILE = 1;
constexpr uint32_t SNN_PER_TILE = DPU_PER_TILE;
constexpr uint32_t SNN_TOTAL = SNN_PER_TILE * MAX_TILES;
constexpr uint32_t AS_PER_TILE = 2;
constexpr uint32_t AS_TOTAL = AS_PER_TILE * MAX_TILES;

constexpr uint32_t MAX_DMA_ENGINES = 2UL;
constexpr uint32_t NUM_METADATA_FEEDERS = 6UL;

// NCE_DPU_HWP_CTRL dtype 2 bit field
enum class HWPStatMode : uint8_t {
    // odu_time_stat, idu_time_stat
    MODE0 = 00,
    // dense_act, sparse_act, dense_wt, sparse_wt when odu done
    MODE1 = 01,
    // dense_wt, sparse_wt, idu_time_stat
    MODE2 = 02,
    // dense_act, sparse_act, idu_time_stat
    MODE3 = 03
};

enum class ActWLType : uint8_t { WL_KERNEL = 0x00, WL_DEBUG = 0x04, WL_UNKNOWN };

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
    U4 = I4,    // The U4 is same as I4 since NCE_DPU_ODU_CFG enums are based on dtype size
    BF16 = FP16,
    OUTPUT_DTYPE_UNKNOWN = 0x0A
};

enum IDUNthw_Ntk {
    IDU_NTHW_NTK_8_8 = 0,
    IDU_NTHW_NTK_4_16 = 1,
    IDU_NTHW_NTK_UNUSED = 2,
    IDU_NTHW_NTK_16_4 = 3,
};

enum ODUGrid { ODU_GRID_4x4, ODU_GRID_16x1, ODU_GRID_4x1 };

enum ODUNthw {
    ODU_NTHW_1,
    ODU_NTHW_4,
    ODU_NTHW_8,
    ODU_NTHW_16,
};

enum MPEGrid { MPE_GRID_4x4, MPE_GRID_16x1 };

} // namespace nn_public
