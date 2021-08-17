#ifndef DESCRIPTOR_H
#define DESCRIPTOR_H
#include <stdint.h>

const uint32_t NCE4_DPU_WORKLOAD_SPECA = 0x0;
const uint32_t NCE4_DPU_WORKLOAD_SPECB = 0x8;
const uint32_t NCE4_DPU_WORKLOAD_SPECC = 0x10;
const uint32_t NCE4_DPU_BLOCKING_AND_STRIDE = 0x18;
const uint32_t NCE4_DPU_PARTITION = 0x20;
const uint32_t NCE4_DPU_OUTER_SCHEDULE = 0x28;
const uint32_t NCE4_DPU_MAP_COLUMN_ARRAY = 0x30;
const uint32_t NCE4_DPU_WORKLOAD_BASE_ADDR0 = 0x38;
const uint32_t NCE4_DPU_WORKLOAD_BASE_ADDR1 = 0x40;
const uint32_t NCE4_DPU_ACTIVATION = 0x48;
const uint32_t NCE4_DPU_WEIGHT = 0x50;
const uint32_t NCE4_DPU_OUTPUT = 0x58;
const uint32_t NCE4_PPE_BIAS = 0x60;
const uint32_t NCE4_PPE_SCALE_CONVERT = 0x68;
const uint32_t NCE4_PPE_CLAMP = 0x70;
const uint32_t NCE4_PPE_FP_BIAS_SCALE = 0x78;
const uint32_t NCE4_PPE_FP_PRELU_SCALE_ADDR = 0x80;

struct workload_speca_reg {
    uint64_t tensor_x_size : 13;
    uint64_t tensor_y_size : 13;
    uint64_t tensor_z_size : 13;
    uint64_t weights : 13;
    uint64_t kernel_x : 4;
    uint64_t kernel_y : 4;
    uint64_t input_act_type : 4;
};

struct workload_specb_reg {
    uint64_t x_start : 13;
    uint64_t y_start : 13;
    uint64_t z_start : 13;
    uint64_t x_end : 13;
    uint64_t top_pad : 4;
    uint64_t bottom_pad : 4;
    uint64_t wgt_type : 4;
};

struct workload_specc_reg {
    uint64_t y_end : 13;
    uint64_t z_end : 13;
    uint64_t left_pad : 4;
    uint64_t right_pad : 4;
    uint64_t pad_value : 10;
    uint64_t output_type : 4;
    uint64_t act_zero_point : 8;
    uint64_t wgt_zero_point : 8;
};

struct blocking_and_stride_reg {
    // Derived from flex_inner
    uint64_t x : 8;   // XB
    uint64_t y : 8;   // YB
    uint64_t ic : 8;  // ICB
    uint64_t oc : 8;  // OCB
    uint64_t fx : 8;  // FXB
    uint64_t fy : 8;  // FYB
    uint64_t dil_conv_x_stride : 4; // Dilated convolution x stride
    uint64_t dil_conv_y_stride : 4; // Dilated convolution y stride
    uint64_t fx_stride : 4;
    uint64_t fy_stride : 4;
};

struct partition_reg {
    // Derived from flex_inner
    uint64_t x : 8;   // XP
    uint64_t y : 8;   // YP
    uint64_t ic : 8;  // ICP
    uint64_t oc : 8;  // OCP
    uint64_t fx : 8;  // FXP
    uint64_t fy : 8;  // FYP
};

struct outer_schedule_reg {
    // Derived from flex_outer
    uint64_t x : 8;   // XP
    uint64_t y : 8;   // YP
    uint64_t ic : 8;  // ICP
    uint64_t oc : 8;  // OCP
    uint64_t fx : 8;  // FXP
    uint64_t fy : 8;  // FYP
    // Encoding of flex_outer_order from the Fathom blob
    // {X:0, Y:1, IC:2, OC:3, FX: 4, FY: 5}
    uint64_t order0 : 3;
    uint64_t order1 : 3;
    uint64_t order2 : 3;
    uint64_t order3 : 3;
};

struct map_column_array_reg {
    uint64_t col_ox : 8;
    uint64_t col_oy : 8;
    uint64_t col_ic : 8;
    uint64_t col_oc : 8;
    uint64_t arr_ox : 8;
    uint64_t arr_oy : 8;
    uint64_t arr_ic : 8;
    uint64_t arr_oc : 8;
};

struct activation_reg {
    uint64_t act_data_addr : 22;
    uint64_t act_sp_addr : 22;
    uint64_t act_base_addr_indx : 1;
    uint64_t is_max_pool : 1;
    uint64_t is_dw_conv : 1;
    uint64_t is_conv_1x1 : 1;
    uint64_t is_conv_NxN : 1;
    uint64_t is_conv_NxNCm : 1;
    uint64_t is_deconv : 1;
    uint64_t is_elop : 1;
    uint64_t sparse_mode : 1;
    uint64_t fp_mode : 1;
};

struct tensor_address {
    uint64_t data_addr : 22;
    uint64_t sp_addr : 22;
    uint64_t sparse_mode : 1;
    uint64_t base_addr_indx : 1;
};

struct ppe4_bias {
    uint64_t ppe_g8_bias_c : 9;
    uint64_t ppe_g8_bias_b : 9;
    uint64_t ppe_g8_bias_a : 9;
    uint64_t ppe_bias : 32;
};

struct ppe_scale_convert {
    uint64_t scale_shift : 6;
    uint64_t scale_round : 2;
    uint64_t scale_mult : 16;
    uint64_t scale_override : 1;
    uint64_t fp_scale_override : 1;
    uint64_t prelu_shift : 5;
    uint64_t prelu_mult : 11;
    uint64_t fp16_ftz : 1;
    uint64_t fp16_clamp : 1;
    uint64_t i32_convert : 1;
    uint64_t fp_convert : 3;
    uint64_t fp_bypass : 1;
    uint64_t bf16_round : 1;
    uint64_t fp_prelu_en : 1;
};

struct ppe4_clamp {
    uint64_t lclamp : 32;
    uint64_t hclamp : 32;
};

struct ppe_fp_bias_scale {
    uint64_t bias : 32;
    uint64_t scale : 32;
};

struct ppe4_fp_prelu_scale_addr {
    uint64_t fp_scale : 32;
    uint64_t bias_scale_addr : 22;
    uint64_t bias_scale_addr_indx : 1;
};

typedef struct {
    struct workload_speca_reg workload_speca;
    struct workload_specb_reg workload_specb;
    struct workload_specc_reg workload_specc;
    struct blocking_and_stride_reg blocking_and_stride;
    struct partition_reg partition;
    struct outer_schedule_reg outer;
    struct map_column_array_reg map_col_arr;
    uint64_t workload_base_addr0;
    uint64_t workload_base_addr1;
    struct activation_reg activation;
    struct tensor_address weight_addr;
    struct tensor_address output_addr;
    struct ppe4_bias ppe_bias;
    struct ppe_scale_convert ppe_scale_conv;
    struct ppe4_clamp ppe_clamp;
    struct ppe_fp_bias_scale fp_bias_scale;
    struct ppe4_fp_prelu_scale_addr ppe_fp_prelu_scale_addr;
}cfg_flexnn_dpu_description;

struct flexnn_job_header {
    uint64_t p_dpu_register_descriptor : 32;
    uint64_t consumer_barrier_mask : 8;
    uint64_t producer_barrier_mask : 8;
    uint64_t column_regcnt : 8;
    uint64_t act_unpacker_regcnt : 8;
    uint64_t wgt_unpacker_regcnt : 8;
    uint64_t drain_regcnt : 8;
    uint64_t total_regcnt : 8;
    uint64_t next_job_pointer : 32;
};

typedef struct {
    struct flexnn_job_header header;
    cfg_flexnn_dpu_description register_descriptor;
}dpu_job_descriptor;

static const uint32_t DPU4_DESCRIPTOR_SIZE = ((sizeof(dpu_job_descriptor) + 7) >> 3) << 3;

typedef struct {
    uint64_t shv_data_address : 32;
    uint64_t shv_kernel_address : 32;
    uint64_t shv_pre_address : 32;
    uint64_t aba_pointer :32;
    uint64_t wait_barrier_mask;
    uint64_t update_barrier_mask;
    uint64_t kernel_size : 32;
    uint64_t data_size : 24;
    uint64_t kernel_arg_count : 8;
    uint64_t kernel_arg_address : 32;
    uint64_t job_completed_pointer : 32;
}shv_job_header;

#endif