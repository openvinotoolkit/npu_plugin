//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#ifndef VPUX_COMPILER_DIALECT_VPU37XX_TYPES
#define VPUX_COMPILER_DIALECT_VPU37XX_TYPES

include "vpux/compiler/dialect/VPURegMapped/dialect.td"
include "vpux/compiler/dialect/VPURegMapped/types.td"
include "vpux/compiler/dialect/VPU37XX/dialect.td"

//
// RegField Specialization
//

class VPU37XX_RegFieldWrapper<string name = "", int width = 0, int pos = 0, string dataType = "UINT"> :
    VPURegMapped_RegFieldWrapper<VPU37XX_Dialect, name, width, pos, dataType> {

    }

//
// Register Specialization
//

class VPU37XX_RegisterWrapper<string name = "", int size = 0, int address = 0x00, list<string> regFieldTypes = [], int resetValue = 0x0, bit allowOverlap = 0> :
    VPURegMapped_RegisterWrapper<VPU37XX_Dialect, "vpux::VPU37XX::", name, size, address, regFieldTypes, resetValue, allowOverlap> {

}

//
// RegMapped Specialization
//

class VPU37XX_RegMappedWrapper<string name = "", list<string> regsTypes = []> :
    VPURegMapped_RegMappedWrapper<VPU37XX_Dialect, "vpux::VPU37XX::", name, regsTypes> {

}

//
// DPUInvariantRegisters
//

// se_sp_addr[4]
def bf_DpuInvSeAddr : VPU37XX_RegFieldWrapper<"se_addr", 32, 0>;
def bf_DpuInvSparsityAddr : VPU37XX_RegFieldWrapper<"sparsity_addr", 32, 32>;

def reg_SeSpAddr0 : VPU37XX_RegisterWrapper<"se_sp_addr_0", 64, 0x0, ["se_addr", "sparsity_addr"]>;
def reg_SeSpAddr1 : VPU37XX_RegisterWrapper<"se_sp_addr_1", 64, 0x8, ["se_addr", "sparsity_addr"]>;
def reg_SeSpAddr2 : VPU37XX_RegisterWrapper<"se_sp_addr_2", 64, 0x10, ["se_addr", "sparsity_addr"]>;
def reg_SeSpAddr3 : VPU37XX_RegisterWrapper<"se_sp_addr_3", 64, 0x18, ["se_addr", "sparsity_addr"]>;

// se_sp_size[3]
def bf_DpuInvSpSegSize : VPU37XX_RegFieldWrapper<"sp_seg_size", 14, 0>;
def bf_DpuInvSeSegSize : VPU37XX_RegFieldWrapper<"se_seg_size", 18, 14>;

def reg_SeSpSize0 : VPU37XX_RegisterWrapper<"se_sp_size_0", 64, 0x20, ["sp_seg_size", "se_seg_size"]>;
def reg_SeSpSize1 : VPU37XX_RegisterWrapper<"se_sp_size_1", 64, 0x24, ["sp_seg_size", "se_seg_size"]>;
def reg_SeSpSize2 : VPU37XX_RegisterWrapper<"se_sp_size_2", 64, 0x28, ["sp_seg_size", "se_seg_size"]>;

// z_config
def bf_SeZSplit : VPU37XX_RegFieldWrapper<"se_z_split", 4, 0>;
def bf_NumSesInZDir : VPU37XX_RegFieldWrapper<"num_ses_in_z_dir", 9, 4>;
def bf_CmSpPattern : VPU37XX_RegFieldWrapper<"cm_sp_pattern", 16, 13>;
def bf_AddrFormatSel : VPU37XX_RegFieldWrapper<"addr_format_sel", 1, 31>;

def reg_ZConfig : VPU37XX_RegisterWrapper<"z_config", 32, 0x2C,
    ["se_z_split", "num_ses_in_z_dir", "cm_sp_pattern", "addr_format_sel"]>;

// kernel_pad_cfg
def bf_KernelY : VPU37XX_RegFieldWrapper<"kernel_y", 4, 5>;
def bf_KernelX : VPU37XX_RegFieldWrapper<"kernel_x", 4, 9>;
def bf_WtPltCfg : VPU37XX_RegFieldWrapper<"wt_plt_cfg", 2, 13>;
def bf_ActDense : VPU37XX_RegFieldWrapper<"act_dense", 1, 15>;
def bf_WtDense : VPU37XX_RegFieldWrapper<"wt_dense", 1, 16>;
def bf_StrideYEn : VPU37XX_RegFieldWrapper<"stride_y_en", 1, 17>;
def bf_StrideY : VPU37XX_RegFieldWrapper<"stride_y", 3, 18>;
def bf_DynamicBwEn : VPU37XX_RegFieldWrapper<"dynamic_bw_en", 1, 21>;
def bf_DwWtSpIns : VPU37XX_RegFieldWrapper<"dw_wt_sp_ins", 1, 22>;
def bf_Layer1WtSpIns : VPU37XX_RegFieldWrapper<"layer1_wt_sp_ins", 1, 23>;
def bf_Layer1CmpEn : VPU37XX_RegFieldWrapper<"layer1_cmp_en", 1, 24>;
def bf_PoolOptEn : VPU37XX_RegFieldWrapper<"pool_opt_en", 1, 25>;
def bf_SpSeTblSegment : VPU37XX_RegFieldWrapper<"sp_se_tbl_segment", 1, 29>;
def bf_RstCtxt : VPU37XX_RegFieldWrapper<"rst_ctxt", 1, 30>;

// kernel_pad_cfg
def reg_KernelPadCfg : VPU37XX_RegisterWrapper<"kernel_pad_cfg", 32, 0x30,
    ["kernel_y", "kernel_x", "wt_plt_cfg", "act_dense", "wt_dense", "stride_y_en", "stride_y", "dynamic_bw_en",
    "dw_wt_sp_ins", "layer1_wt_sp_ins", "layer1_cmp_en", "pool_opt_en", "sp_se_tbl_segment", "rst_ctxt"]>;

// weight_size_placeholder
def bf_WeightSizePlaceholder : VPU37XX_RegFieldWrapper<"weight_size_placeholder", 32, 0>;

def reg_WeightSizePlaceholder : VPU37XX_RegisterWrapper<"weight_size_placeholder", 32, 0x34, ["weight_size_placeholder"]>;

// weight_num_placeholder
def bf_WeightNumPlaceholder : VPU37XX_RegFieldWrapper<"weight_num_placeholder", 32, 0>;

def reg_WeightNumPlaceholder : VPU37XX_RegisterWrapper<"weight_num_placeholder", 32, 0x38, ["weight_num_placeholder"]>;

// weight_start
def bf_WeightStart : VPU37XX_RegFieldWrapper<"weight_start", 32, 0>;

def reg_WeightStart : VPU37XX_RegisterWrapper<"weight_start", 32, 0x3C, ["weight_start"]>;

// tensor_size0
def bf_TensorSizeX : VPU37XX_RegFieldWrapper<"tensor_size_x", 14, 0>;

def bf_TensorSizeY : VPU37XX_RegFieldWrapper<"tensor_size_y", 14, 14>;

def reg_TensorSize0 : VPU37XX_RegisterWrapper<"tensor_size0", 32, 0x40, ["tensor_size_x", "tensor_size_y"]>;

// tensor_size1
def bf_TensorSizeZ : VPU37XX_RegFieldWrapper<"tensor_size_z", 14, 0>;

def reg_TensorSize1 : VPU37XX_RegisterWrapper<"tensor_size1", 32, 0x44, ["tensor_size_z"]>;

// tensor_start
def bf_TensorStart : VPU37XX_RegFieldWrapper<"tensor_start", 32, 0>;

def reg_TensorStart : VPU37XX_RegisterWrapper<"tensor_start", 32, 0x48, ["tensor_start"]>;

// tensor_mode
def bf_Wmode : VPU37XX_RegFieldWrapper<"wmode", 4, 0>;
def bf_Amode : VPU37XX_RegFieldWrapper<"amode", 4, 4>;
def bf_Stride : VPU37XX_RegFieldWrapper<"stride", 3, 8>;
def bf_ZmInput : VPU37XX_RegFieldWrapper<"zm_input", 1, 11>;
def bf_DwInput : VPU37XX_RegFieldWrapper<"dw_input", 1, 12>;
def bf_WorkloadOperation : VPU37XX_RegFieldWrapper<"workload_operation", 2, 14>;

def reg_TensorMode : VPU37XX_RegisterWrapper<"tensor_mode", 32, 0x4c,
    ["wmode", "amode", "stride", "zm_input", "dw_input", "workload_operation"]>;


// elops_sparsity_addr
def bf_ElopsSparityAddr : VPU37XX_RegFieldWrapper<"elops_sparsity_addr", 32, 0>;

def reg_ElopsSparityAddr : VPU37XX_RegisterWrapper<"elops_sparsity_addr", 32, 0x50, ["elops_sparsity_addr"]>;

// elops_se_addr
def bf_ElopsSeAddr : VPU37XX_RegFieldWrapper<"elops_se_addr", 32, 0>;

def reg_ElopsSeAddr : VPU37XX_RegisterWrapper<"elops_se_addr", 32, 0x54, ["elops_se_addr"]>;

// elops_wload
def bf_ElopWload : VPU37XX_RegFieldWrapper<"elop_wload", 1, 0>;
def bf_ElopWloadType : VPU37XX_RegFieldWrapper<"elop_wload_type", 1, 3>;
def bf_PoolWtData : VPU37XX_RegFieldWrapper<"pool_wt_data", 16, 4>;
def bf_PoolWtRdDis : VPU37XX_RegFieldWrapper<"pool_wt_rd_dis", 1, 26>;

def reg_ElopsWload : VPU37XX_RegisterWrapper<"elops_wload", 32, 0x58,
    ["elop_wload", "elop_wload_type", "pool_wt_data", "pool_wt_rd_dis"], 0x0>;


// act0_offset [4]
def bf_Adr0Offset : VPU37XX_RegFieldWrapper<"act_offset", 32, 0>;

def reg_Act0Offset : VPU37XX_RegisterWrapper<"act_offset0", 32, 0x5C, ["act_offset"]>;
def reg_Act1Offset : VPU37XX_RegisterWrapper<"act_offset1", 32, 0x60, ["act_offset"]>;
def reg_Act2Offset : VPU37XX_RegisterWrapper<"act_offset2", 32, 0x64, ["act_offset"]>;
def reg_Act3Offset : VPU37XX_RegisterWrapper<"act_offset3", 32, 0x68, ["act_offset"]>;

// base_offset_a
def bf_BaseOffsetA : VPU37XX_RegFieldWrapper<"base_offset_a", 32, 0>;
def reg_BaseOffsetA : VPU37XX_RegisterWrapper<"base_offset_a", 32, 0x6C, ["base_offset_a"]>;

// base_offset_b
def bf_BaseOffsetB : VPU37XX_RegFieldWrapper<"base_offset_b", 32, 0>;
def reg_BaseOffsetB : VPU37XX_RegisterWrapper<"base_offset_b", 32, 0x70, ["base_offset_b"]>;

// wt_offset
def bf_WtOffset : VPU37XX_RegFieldWrapper<"wt_offset", 32, 0>;
def reg_WtOffset : VPU37XX_RegisterWrapper<"wt_offset", 32, 0x74, ["wt_offset"]>;

// odu_cfg
def bf_Dtype : VPU37XX_RegFieldWrapper<"dtype", 4, 0>;
def bf_SpValue : VPU37XX_RegFieldWrapper<"sp_value", 8, 5>;
def bf_SpOutEn : VPU37XX_RegFieldWrapper<"sp_out_en", 1, 13>;
def bf_WriteSp : VPU37XX_RegFieldWrapper<"write_sp", 1, 15>;
def bf_WritePt : VPU37XX_RegFieldWrapper<"write_pt", 1, 16>;
def bf_WriteAc : VPU37XX_RegFieldWrapper<"write_ac", 1, 17>;
def bf_Mode : VPU37XX_RegFieldWrapper<"mode", 2, 18>;
def bf_Grid : VPU37XX_RegFieldWrapper<"grid", 1, 20>;
def bf_SwizzleKey : VPU37XX_RegFieldWrapper<"swizzle_key", 3, 21>;
def bf_Nthw : VPU37XX_RegFieldWrapper<"nthw", 2, 25>;
def bf_Permutation : VPU37XX_RegFieldWrapper<"permutation", 3, 27>;
def bf_WcbStallAvoidance : VPU37XX_RegFieldWrapper<"debug_mode", 1, 30>;

def reg_OduCfg : VPU37XX_RegisterWrapper<"odu_cfg", 32, 0x78,
    ["dtype", "sp_value", "sp_out_en", "write_sp", "write_pt", "write_ac", "mode", "grid",
    "swizzle_key", "nthw", "permutation", "debug_mode"]>;

// odu_be_size
def bf_OduBeSize : VPU37XX_RegFieldWrapper<"odu_be_size", 32, 0>;

def reg_OduBeSize : VPU37XX_RegisterWrapper<"odu_be_size", 32, 0x7C, ["odu_be_size"]>;

// odu_be_cnt
def bf_OduBeCnt : VPU37XX_RegFieldWrapper<"odu_be_cnt", 32, 0>;

def reg_OduBeCnt : VPU37XX_RegisterWrapper<"odu_be_cnt", 32, 0x80, ["odu_be_cnt"]>;

// odu_se_size
def bf_SeSize : VPU37XX_RegFieldWrapper<"se_size", 32, 0>;
def reg_SeSize : VPU37XX_RegisterWrapper<"se_size", 32, 0x84, ["se_size"]>;

// te_dim0
def bf_TeDimY : VPU37XX_RegFieldWrapper<"te_dim_y", 13, 0>;
def bf_TeDimZ : VPU37XX_RegFieldWrapper<"te_dim_z", 13, 13>;

def reg_TeDim0 : VPU37XX_RegisterWrapper<"te_dim0", 32, 0x88, ["te_dim_y", "te_dim_z"]>;

// te_dim1
def bf_TeDimX : VPU37XX_RegFieldWrapper<"te_dim_x", 13, 0>;

def reg_TeDim1 : VPU37XX_RegisterWrapper<"te_dim1", 32, 0x8C, ["te_dim_x"]>;

// pt_base
def bf_PtBase : VPU37XX_RegFieldWrapper<"pt_base", 32, 0>;

def reg_PtBase : VPU37XX_RegisterWrapper<"pt_base", 32, 0x90, ["pt_base"]>;

// sp_base
def bf_SpBase : VPU37XX_RegFieldWrapper<"sp_base", 32, 0>;

def reg_SpBase : VPU37XX_RegisterWrapper<"sp_base", 32, 0x94, ["sp_base"]>;

// base_ptr_a
def bf_BasePtrA : VPU37XX_RegFieldWrapper<"base_ptr_a", 32, 0>;

def reg_BasePtrA : VPU37XX_RegisterWrapper<"base_ptr_a", 32, 0x98, ["base_ptr_a"]>;

// base_ptr_b
def bf_BasePtrB : VPU37XX_RegFieldWrapper<"base_ptr_b", 32, 0>;

def reg_BasePtrB : VPU37XX_RegisterWrapper<"base_ptr_b", 32, 0x9C, ["base_ptr_b"]>;

// base_adr
def bf_BaseAdr : VPU37XX_RegFieldWrapper<"base_adr", 32, 0>;

def reg_BaseAdr : VPU37XX_RegisterWrapper<"base_adr0", 32, 0xa0, ["base_adr"]>;
def reg_BaseAdr1 : VPU37XX_RegisterWrapper<"base_adr1", 32, 0xa4, ["base_adr"]>;
def reg_BaseAdr2 : VPU37XX_RegisterWrapper<"base_adr2", 32, 0xa8, ["base_adr"]>;
def reg_BaseAdr3 : VPU37XX_RegisterWrapper<"base_adr3", 32, 0xaC, ["base_adr"]>;

// odu_cast
def bf_OduCastEn : VPU37XX_RegFieldWrapper<"cast_enable", 1, 0>;
def bf_OduCastOffset : VPU37XX_RegFieldWrapper<"cast_offset", 28, 4>;

def reg_OduCast0 : VPU37XX_RegisterWrapper<"odu_cast0", 32, 0xb0, ["cast_enable", "cast_offset"]>;
def reg_OduCast1 : VPU37XX_RegisterWrapper<"odu_cast1", 32, 0xb4, ["cast_enable", "cast_offset"]>;
def reg_OduCast2 : VPU37XX_RegisterWrapper<"odu_cast2", 32, 0xb8, ["cast_enable", "cast_offset"]>;
def reg_OduCast3 : VPU37XX_RegisterWrapper<"odu_cast3", 32, 0xbC, ["cast_enable", "cast_offset"]>;

// mpe_cfg
def bf_MpeWtbias : VPU37XX_RegFieldWrapper<"mpe_wtbias", 8, 0, "UINT">;
def bf_MpeActbias : VPU37XX_RegFieldWrapper<"mpe_actbias", 8, 8, "UINT">;
def bf_MpeDaz : VPU37XX_RegFieldWrapper<"mpe_daz", 1, 22>;

def reg_MpeCfg : VPU37XX_RegisterWrapper<"mpe_cfg", 32, 0xBC,
    ["mpe_wtbias", "mpe_actbias", "mpe_daz"]>;

// elop_scale
def bf_ElopScaleB : VPU37XX_RegFieldWrapper<"elop_scale_b", 16, 0>;
def bf_ElopScaleA : VPU37XX_RegFieldWrapper<"elop_scale_a", 16, 16>;

def reg_ElopScale : VPU37XX_RegisterWrapper<"elop_scale", 32, 0xC4, ["elop_scale_b", "elop_scale_a"]>;

// ppe_cfg
def bf_PpeG8BiasC : VPU37XX_RegFieldWrapper<"ppe_g8_bias_c", 9, 0, "SINT">;

def reg_PpeCfg : VPU37XX_RegisterWrapper<"ppe_cfg", 32, 0xC8, ["ppe_g8_bias_c"]>;

// ppe_bias
def bf_PpeBias : VPU37XX_RegFieldWrapper<"ppe_bias", 32, 0, "SINT">;

def reg_PpeBias : VPU37XX_RegisterWrapper<"ppe_bias", 32, 0xCC, ["ppe_bias"]>;

// ppe_scale
def bf_PpeScaleShift : VPU37XX_RegFieldWrapper<"ppe_scale_shift", 6, 2>;
def bf_PpeScaleRound : VPU37XX_RegFieldWrapper<"ppe_scale_round", 2, 10>;
def bf_PpeScaleMult : VPU37XX_RegFieldWrapper<"ppe_scale_mult", 16, 16, "SINT">;

def reg_PpeScale : VPU37XX_RegisterWrapper<"ppe_scale", 32, 0xD0, ["ppe_scale_shift", "ppe_scale_round", "ppe_scale_mult"], 0x00000c00>;

// ppe_scale_ctrl
def bf_PpeScaleOverride : VPU37XX_RegFieldWrapper<"ppe_scale_override", 1, 0>;
def bf_PpeFpScaleOverride : VPU37XX_RegFieldWrapper<"ppe_fp_scale_override", 1, 1>;

def reg_PpeScaleCtrl : VPU37XX_RegisterWrapper<"ppe_scale_ctrl", 32, 0xD4, ["ppe_scale_override", "ppe_fp_scale_override"], 0x0>;


// ppe_prelu
def bf_PpePreluShift : VPU37XX_RegFieldWrapper<"ppe_prelu_shift", 5, 8>;
def bf_PpePreluMult : VPU37XX_RegFieldWrapper<"ppe_prelu_mult", 11, 16>;

def reg_PpePrelu : VPU37XX_RegisterWrapper<"ppe_prelu", 32, 0xD8, ["ppe_prelu_shift", "ppe_prelu_mult"], 0x00010000>;

// ppe_scale_hclamp
def bf_PpeScaleHclamp : VPU37XX_RegFieldWrapper<"ppe_scale_hclamp", 32, 0, "SINT">;
def reg_PpeScaleHclamp : VPU37XX_RegisterWrapper<"ppe_scale_hclamp", 32, 0xE4, ["ppe_scale_hclamp"], 0x7fffffff>;

// ppe_scale_lclamp
def bf_PpeScaleLclamp : VPU37XX_RegFieldWrapper<"ppe_scale_lclamp", 32, 0, "SINT">;
def reg_PpeScaleLclamp : VPU37XX_RegisterWrapper<"ppe_scale_lclamp", 32, 0xE8, ["ppe_scale_lclamp"], 0x80000000>;

// ppe_misc
def bf_PpeFp16Ftz : VPU37XX_RegFieldWrapper<"ppe_fp16_ftz", 1, 6>;
def bf_PpeFp16Clamp : VPU37XX_RegFieldWrapper<"ppe_fp16_clamp", 1, 7>;
def bf_PpeI32Convert : VPU37XX_RegFieldWrapper<"ppe_i32_convert", 2, 8>;
def reg_PpeMisc : VPU37XX_RegisterWrapper<"ppe_misc", 32, 0xF0, ["ppe_fp16_ftz", "ppe_fp16_clamp", "ppe_i32_convert"]>;

// ppe_fp_bias
def bf_PpeFpBias : VPU37XX_RegFieldWrapper<"ppe_fp_bias", 32, 0, "FP">;
def reg_PpeFpBias : VPU37XX_RegisterWrapper<"ppe_fp_bias", 32, 0xF4, ["ppe_fp_bias"]>;

// ppe_fp_scale
def bf_PpeFpScale : VPU37XX_RegFieldWrapper<"ppe_fp_scale", 32, 0, "FP">;
def reg_PpeFpScale : VPU37XX_RegisterWrapper<"ppe_fp_scale", 32, 0xF8, ["ppe_fp_scale"]>;

// ppe_fp_prelu
def bf_PpeFpPrelu : VPU37XX_RegFieldWrapper<"ppe_fp_prelu", 32, 0, "FP">;
def reg_PpeFpPrelu : VPU37XX_RegisterWrapper<"ppe_fp_prelu", 32, 0xFC, ["ppe_fp_prelu"]>;

// ppe_fp_cfg
def bf_PpeFpConvert : VPU37XX_RegFieldWrapper<"ppe_fp_convert", 3, 0>;
def bf_PpeFpBypass : VPU37XX_RegFieldWrapper<"ppe_fp_bypass", 1, 3>;
def bf_PpeBf16Round : VPU37XX_RegFieldWrapper<"ppe_bf16_round", 1, 4>;
def bf_PpeFpPreluEn : VPU37XX_RegFieldWrapper<"ppe_fp_prelu_en", 1, 5>;
//  Note: ppe prelu needs to be off by default, only turned on explicitly
def reg_PpeFpCfg : VPU37XX_RegisterWrapper<"ppe_fp_cfg", 32, 0x100, ["ppe_fp_convert", "ppe_fp_bypass", "ppe_bf16_round", "ppe_fp_prelu_en"], 0x00000008>;

//  hwp_cmx_base_offset_
def bf_HwpCmxBaseOffset : VPU37XX_RegFieldWrapper<"hwp_cmx_base_offset_", 32, 0>;
def reg_HwpCmxBaseOffset : VPU37XX_RegisterWrapper<"hwp_cmx_base_offset_", 32, 0x104, ["hwp_cmx_base_offset_"]>;

// barriers_
def bf_BarrierWaitMask : VPU37XX_RegFieldWrapper<"barriers_wait_mask_", 64, 0>;
def reg_BarrierWaitMask : VPU37XX_RegisterWrapper<"barriers_wait_mask_", 64, 0x108, ["barriers_wait_mask_"]>;

def bf_BarrierWaitPost : VPU37XX_RegFieldWrapper<"barriers_post_mask_", 64, 0>;
def reg_BarrierWaitPost : VPU37XX_RegisterWrapper<"barriers_post_mask_", 64, 0x110, ["barriers_post_mask_"]>;

def bf_BarrierGroup : VPU37XX_RegFieldWrapper<"group_", 8, 0>;
def bf_BarrierMask : VPU37XX_RegFieldWrapper<"mask_", 8, 8>;
def reg_BarrierGroupMask : VPU37XX_RegisterWrapper<"barriers_group_mask_", 64, 0x118, ["group_", "mask_"]>;

// barriers_sched_
def bf_StartAfter : VPU37XX_RegFieldWrapper<"start_after_", 16, 0>;
def bf_CleanAfter : VPU37XX_RegFieldWrapper<"clean_after_", 16, 16>;
def reg_BarriersSchedulingCfg : VPU37XX_RegisterWrapper<"barriers_sched_", 32, 0x120, ["start_after_", "clean_after_"]>;

// variant_count_
def bf_VariantCnt : VPU37XX_RegFieldWrapper<"variant_count_", 16, 0>;
def reg_VariantCnt : VPU37XX_RegisterWrapper<"variant_count_", 16, 0x124, ["variant_count_"]>;

// cluster_
def bf_Cluster : VPU37XX_RegFieldWrapper<"cluster_", 8, 0>;
def reg_ClusterInvariant : VPU37XX_RegisterWrapper<"cluster_invariant_", 8, 0x126, ["cluster_"]>;

// is_cont_conv_
def bf_IsContConv : VPU37XX_RegFieldWrapper<"is_cont_conv_", 8, 0>;
def reg_IsContConv : VPU37XX_RegisterWrapper<"is_cont_conv_", 8, 0x127, ["is_cont_conv_"]>;

// DPUInvariantRegister
def reg_mapped_DpuInvariantRegister : VPU37XX_RegMappedWrapper<"DpuInvariantRegister",
    ["se_sp_addr_0", "se_sp_addr_1", "se_sp_addr_2", "se_sp_addr_3",
    "se_sp_size_0", "se_sp_size_1", "se_sp_size_2",
    "z_config",
    "kernel_pad_cfg",
    "weight_size_placeholder", "weight_num_placeholder",
    "weight_start",
    "tensor_size0", "tensor_size1",
    "tensor_start",
    "tensor_mode",
    "elops_sparsity_addr",
    "elops_se_addr",
    "elops_wload",
    "tensor_mode",
    "act_offset0", "act_offset1", "act_offset2", "act_offset3",
    "base_offset_a", "base_offset_b",
    "wt_offset",
    "odu_cfg",
    "odu_be_size",
    "odu_be_cnt",
    "se_size",
    "te_dim0", "te_dim1",
    "pt_base",
    "sp_base",
    "base_ptr_a", "base_ptr_b",
    "base_adr0", "base_adr1", "base_adr2", "base_adr3",
    "odu_cast0", "odu_cast1", "odu_cast2", "odu_cast3",
    "mpe_cfg",
    "elop_scale",
    "ppe_cfg", "ppe_bias", "ppe_scale", "ppe_scale_ctrl", "ppe_prelu",
    "ppe_scale_hclamp", "ppe_scale_lclamp",
    "ppe_misc",
    "ppe_fp_bias", "ppe_fp_scale", "ppe_fp_prelu", "ppe_fp_cfg",
    "hwp_cmx_base_offset_",
     "barriers_wait_mask_", "barriers_post_mask_", "barriers_group_mask_",
     "barriers_sched_", "variant_count_", "cluster_invariant_", "is_cont_conv_"
    ]>;

//
// DPUVariantRegisters
//

// workload_size0
def bf_WorkloadSizeX : VPU37XX_RegFieldWrapper<"workload_size_x", 14, 0>;
def bf_WorkloadSizeY : VPU37XX_RegFieldWrapper<"workload_size_y", 14, 14>;

def reg_WorkloadSize0 : VPU37XX_RegisterWrapper<"workload_size0", 32, 0x0, ["workload_size_x", "workload_size_y"]>;

// workload_size1
def bf_WorkloadSizeZ : VPU37XX_RegFieldWrapper<"workload_size_z", 14, 0>;
def bf_PadCountUp : VPU37XX_RegFieldWrapper<"pad_count_up", 3, 14>;
def bf_PadCountLeft : VPU37XX_RegFieldWrapper<"pad_count_left", 3, 17>;
def bf_PadCountDown : VPU37XX_RegFieldWrapper<"pad_count_down", 3, 20>;
def bf_PadCountRight : VPU37XX_RegFieldWrapper<"pad_count_right", 3, 23>;

def reg_WorkloadSize1 : VPU37XX_RegisterWrapper<"workload_size1", 32, 0x4,
    ["workload_size_z", "pad_count_up", "pad_count_left", "pad_count_down", "pad_count_right"]>;

// workload_start0
def bf_WorkloadStartX : VPU37XX_RegFieldWrapper<"workload_start_x", 14, 0>;
def bf_WorkloadStartY : VPU37XX_RegFieldWrapper<"workload_start_y", 14, 14>;
def reg_WorkloadStart0 : VPU37XX_RegisterWrapper<"workload_start0", 32, 0x8, ["workload_start_x", "workload_start_y"]>;

// workload_start1
def bf_WorkloadStartZ : VPU37XX_RegFieldWrapper<"workload_start_z", 14, 0>;
def reg_WorkloadStart1 : VPU37XX_RegisterWrapper<"workload_start1", 32, 0xC, ["workload_start_z"]>;

// offset_addr
def bf_NthwNtk : VPU37XX_RegFieldWrapper<"nthw_ntk", 2, 0>;
def bf_BinCfg : VPU37XX_RegFieldWrapper<"bin_cfg", 1, 2>;
def bf_ConvCond : VPU37XX_RegFieldWrapper<"conv_cond", 1, 3>;
def bf_DenseSe : VPU37XX_RegFieldWrapper<"dense_se", 1, 4>;
def bf_IdxQuad : VPU37XX_RegFieldWrapper<"idx_quad", 1, 5>;  // unused
def bf_SwizzleKeyOffsetAddr : VPU37XX_RegFieldWrapper<"swizzle_key_offset", 3, 6>;
def bf_IduMrmClkEn : VPU37XX_RegFieldWrapper<"idu_mrm_clk_en", 1, 9>;
def bf_OduClkEn : VPU37XX_RegFieldWrapper<"odu_clk_en", 1, 10>;
def bf_MpeClkEn : VPU37XX_RegFieldWrapper<"mpe_clk_en", 1, 11>;
def bf_PpeClkEn : VPU37XX_RegFieldWrapper<"ppe_clk_en", 1, 12>;
def bf_OduStatEn : VPU37XX_RegFieldWrapper<"odu_stat_en", 1, 13>;
def bf_IduStatEn : VPU37XX_RegFieldWrapper<"idu_stat_en", 1, 14>;
def bf_OduStatClrMode : VPU37XX_RegFieldWrapper<"odu_stat_clr_mode", 1, 16>;
def bf_IduStatClrMode : VPU37XX_RegFieldWrapper<"idu_stat_clr_mode", 1, 17>;
def bf_ShaveL2CacheEn : VPU37XX_RegFieldWrapper<"shave_l2_cache_en", 1, 19>;
def bf_IduDbgEn : VPU37XX_RegFieldWrapper<"idu_dbg_en", 2, 20>;
def bf_WtSwizzleKey : VPU37XX_RegFieldWrapper<"wt_swizzle_key", 3, 27>;
def bf_WtSwizzleSel : VPU37XX_RegFieldWrapper<"wt_swizzle_sel", 1, 30>;
//  Note: wt_swizzle_sel needs to be on by default to match NNRT GF behaviour
def reg_OffsetAddr : VPU37XX_RegisterWrapper<"offset_addr", 32, 0x10,
    ["nthw_ntk", "bin_cfg", "conv_cond", "dense_se", "swizzle_key_offset", "idu_mrm_clk_en", "odu_clk_en",
    "mpe_clk_en", "ppe_clk_en", "odu_stat_en", "idu_stat_en", "odu_stat_clr_mode", "idu_stat_clr_mode", "shave_l2_cache_en",
    "idu_dbg_en", "wt_swizzle_key", "wt_swizzle_sel"], 0x40000000>;

// te_end0
def bf_TeEndY : VPU37XX_RegFieldWrapper<"te_end_y", 13, 0>;
def bf_TeEndZ : VPU37XX_RegFieldWrapper<"te_end_z", 13, 13>;
def reg_TeEnd0 : VPU37XX_RegisterWrapper<"te_end0", 32, 0x14, ["te_end_y", "te_end_z"]>;

// te_end1
def bf_TeEndX : VPU37XX_RegFieldWrapper<"te_end_x", 13, 0>;
def reg_TeEnd1 : VPU37XX_RegisterWrapper<"te_end1", 32, 0x18, ["te_end_x"]>;

// te_beg0
def bf_TeBegY : VPU37XX_RegFieldWrapper<"te_beg_y", 13, 0>;
def bf_TeBegZ : VPU37XX_RegFieldWrapper<"te_beg_z", 13, 13>;
def reg_TeBeg0 : VPU37XX_RegisterWrapper<"te_beg0", 32, 0x1C, ["te_beg_y", "te_beg_z"]>;

// te_beg1
def bf_TeBegX : VPU37XX_RegFieldWrapper<"te_beg_x", 13, 0>;
def reg_TeBeg1 : VPU37XX_RegisterWrapper<"te_beg1", 32, 0x20, ["te_beg_x"]>;

// weight_size
def bf_WeightSize : VPU37XX_RegFieldWrapper<"weight_size", 32, 0>;
def reg_WeightSize : VPU37XX_RegisterWrapper<"weight_size", 32, 0x24, ["weight_size"]>;

// weight_num
def bf_WeighNum : VPU37XX_RegFieldWrapper<"weight_num", 32, 0>;
def reg_WeightNum : VPU37XX_RegisterWrapper<"weight_num", 32, 0x28, ["weight_num"]>;

// invariant_
def bf_Invariant : VPU37XX_RegFieldWrapper<"invariant_", 32, 0>;
def reg_Invariant : VPU37XX_RegisterWrapper<"invariant_", 32, 0x2C, ["invariant_"]>;

def bf_InvariantIndex : VPU37XX_RegFieldWrapper<"invariant_index_", 32, 0>;
def reg_InvariantIndex : VPU37XX_RegisterWrapper<"invariant_index_", 32, 0x30, ["invariant_index_"]>;

def bf_WeightTableOffset : VPU37XX_RegFieldWrapper<"weight_table_offset_", 32, 0>;
def reg_WeightTableOffset : VPU37XX_RegisterWrapper<"weight_table_offset_", 32, 0x34, ["weight_table_offset_"]>;

def bf_WloadId : VPU37XX_RegFieldWrapper<"wload_id_", 32, 0>;
def reg_WloadId : VPU37XX_RegisterWrapper<"wload_id_", 32, 0x38, ["wload_id_"]>;

def reg_Cluster : VPU37XX_RegisterWrapper<"cluster_variant_", 32, 0x3C, ["cluster_"]>;

// DPUVariantRegister
def reg_mapped_DpuVariantRegister : VPU37XX_RegMappedWrapper<"DpuVariantRegister",
    ["workload_size0", "workload_size1", "workload_start0", "workload_start1", "offset_addr", "te_end0", "te_end1", "te_beg0", "te_beg1",
    "weight_size", "weight_num", "invariant_", "invariant_index_", "weight_table_offset_", "wload_id_", "cluster_variant_"]>;

// DMADescriptor

//
//  dma_64_bit_word_0
//

def bf_dma_link_address : VPU37XX_RegFieldWrapper<"dma_link_address", 37, 0>;                           // Pointer to the next element in linked list
def reg_dma_link_address : VPU37XX_RegisterWrapper<"dma_link_address", 64, 0x0, ["dma_link_address"]>;

def bf_dma_watermark : VPU37XX_RegFieldWrapper<"dma_watermark", 1, 63>;                                  // Watermark to indicate that the job has completed
def reg_dma_watermark : VPU37XX_RegisterWrapper<"dma_watermark", 64, 0x0, ["dma_watermark"]>;

//
//  dma_64_bit_word_1
//

def bf_dma_type : VPU37XX_RegFieldWrapper<"dma_type", 2, 0>; 
def bf_dma_burst_length : VPU37XX_RegFieldWrapper<"dma_burst_length", 8, 2>;
def bf_dma_critical : VPU37XX_RegFieldWrapper<"dma_critical", 1, 10>;
def bf_dma_interrupt_en : VPU37XX_RegFieldWrapper<"dma_interrupt_en", 1, 11>;
def bf_dma_interrupt_trigger : VPU37XX_RegFieldWrapper<"dma_interrupt_trigger", 7, 12>;
def bf_dma_skip_nr : VPU37XX_RegFieldWrapper<"dma_skip_nr", 7, 19>;
def bf_dma_order_forced : VPU37XX_RegFieldWrapper<"dma_order_forced", 1, 26>;
def bf_dma_watermark_en : VPU37XX_RegFieldWrapper<"dma_watermark_en", 1, 27>;
def bf_dma_dec_en : VPU37XX_RegFieldWrapper<"dma_dec_en", 1, 28>;
def bf_dma_barrier_en : VPU37XX_RegFieldWrapper<"dma_barrier_en", 1, 29>;

def reg_dma_cfg_bits : VPU37XX_RegisterWrapper<"dma_cfg_bits", 64, 0x8, 
    ["dma_type", "dma_burst_length", "dma_critical", "dma_interrupt_en", "dma_interrupt_trigger", "dma_skip_nr",
     "dma_order_forced", "dma_watermark_en", "dma_dec_en", "dma_barrier_en"]>;

//
//  dma_64_bit_word_2
//

def bf_dma_src : VPU37XX_RegFieldWrapper<"dma_src", 38, 0>;
def reg_dma_src : VPU37XX_RegisterWrapper<"dma_src", 64, 0x10, ["dma_src"]>;

//
//  dma_64_bit_word_3
//

def bf_dma_dst : VPU37XX_RegFieldWrapper<"dma_dst", 38, 0>;
def reg_dma_dst : VPU37XX_RegisterWrapper<"dma_dst", 64, 0x18, ["dma_dst"]>;

//
//  dma_64_bit_word_4
//

def bf_dma_length : VPU37XX_RegFieldWrapper<"dma_length", 24, 0>;
def reg_dma_length : VPU37XX_RegisterWrapper<"dma_length", 32, 0x20, ["dma_length"]>;

def bf_dma_num_planes : VPU37XX_RegFieldWrapper<"dma_num_planes", 8, 0>;
def reg_dma_num_planes : VPU37XX_RegisterWrapper<"dma_num_planes", 8, 0x24, ["dma_num_planes"]>;

def bf_dma_task_id : VPU37XX_RegFieldWrapper<"dma_task_id", 24, 0>;
def reg_dma_task_id : VPU37XX_RegisterWrapper<"dma_task_id", 24, 0x25, ["dma_task_id"]>;

//
//  dma_64_bit_word_5
//

def bf_dma_src_plane_stride : VPU37XX_RegFieldWrapper<"dma_src_plane_stride", 32, 0>;
def reg_dma_src_plane_stride : VPU37XX_RegisterWrapper<"dma_src_plane_stride", 32, 0x28, ["dma_src_plane_stride"]>;

def bf_dma_dst_plane_stride : VPU37XX_RegFieldWrapper<"dma_dst_plane_stride", 32, 0>;
def reg_dma_dst_plane_stride : VPU37XX_RegisterWrapper<"dma_dst_plane_stride", 32, 0x2C, ["dma_dst_plane_stride"]>;

//
//  dma_64_bit_word_6
//

//  2D case

def bf_dma_attr2d_src_width : VPU37XX_RegFieldWrapper<"dma_attr2d_src_width", 24, 0>;
def reg_dma_attr2d_src_width : VPU37XX_RegisterWrapper<"dma_attr2d_src_width", 32, 0x30, ["dma_attr2d_src_width"]>;

def bf_dma_attr2d_src_stride : VPU37XX_RegFieldWrapper<"dma_attr2d_src_stride", 32, 0>;
def reg_dma_attr2d_src_stride : VPU37XX_RegisterWrapper<"dma_attr2d_src_stride", 32, 0x34, ["dma_attr2d_src_stride"]>;

// 1D case

def bf_dma_barriers_prod_mask : VPU37XX_RegFieldWrapper<"dma_barriers_prod_mask", 64, 0>;
def reg_dma_barriers1d_prod_mask : VPU37XX_RegisterWrapper<"dma_barriers1d_prod_mask", 64, 0x30, ["dma_barriers_prod_mask"]>;

//
//  dma_64_bit_word_7
//

//  2D case

def bf_dma_attr2d_dst_width : VPU37XX_RegFieldWrapper<"dma_attr2d_dst_width", 24, 0>;
def reg_dma_attr2d_dst_width : VPU37XX_RegisterWrapper<"dma_attr2d_dst_width", 32, 0x38, ["dma_attr2d_dst_width"]>;

def bf_dma_attr2d_dst_stride : VPU37XX_RegFieldWrapper<"dma_attr2d_dst_stride", 32, 0>;
def reg_dma_attr2d_dst_stride : VPU37XX_RegisterWrapper<"dma_attr2d_dst_stride", 32, 0x3C, ["dma_attr2d_dst_stride"]>;

// 1D case

def bf_dma_barriers_cons_mask : VPU37XX_RegFieldWrapper<"dma_barriers_cons_mask", 64, 0>;
def reg_dma_barriers1d_cons_mask : VPU37XX_RegisterWrapper<"dma_barriers1d_cons_mask", 64, 0x38, ["dma_barriers_cons_mask"]>;

//
//  dma_64_bit_word_8 (used for 2d case only)
//

def reg_dma_barriers_prod_mask : VPU37XX_RegisterWrapper<"dma_barriers_prod_mask", 64, 0x40, ["dma_barriers_prod_mask"]>;

//
//  dma_64_bit_word_9 (used for 2d case only)
//

def reg_dma_barriers_cons_mask : VPU37XX_RegisterWrapper<"dma_barriers_cons_mask", 64, 0x48, ["dma_barriers_cons_mask"]>;

//
//  VpuTaskSchedulingBarrierConfig
//

def bf_dma_barriers_sched_start_after : VPU37XX_RegFieldWrapper<"dma_barriers_sched_start_after", 16, 0>;
def bf_dma_barriers_sched_clean_after : VPU37XX_RegFieldWrapper<"dma_barriers_sched_clean_after", 16, 16>;
def reg_VpuTaskSchedulingBarrierConfig : VPU37XX_RegisterWrapper<"dma_barriers_sched", 32, 0x50,
    ["dma_barriers_sched_start_after", "dma_barriers_sched_clean_after"]>;

//
//  Pads
//

//def reg_dma_pad : VPU37XX_RegisterWrapper<"dma_pad_0", 352, 0x54, []>;
def reg_dma_pad_0 : VPU37XX_RegisterWrapper<"dma_pad_0", 32, 0x54, []>;
def reg_dma_pad_1 : VPU37XX_RegisterWrapper<"dma_pad_1", 64, 0x58, []>;
def reg_dma_pad_2 : VPU37XX_RegisterWrapper<"dma_pad_2", 64, 0x60, []>;
def reg_dma_pad_3 : VPU37XX_RegisterWrapper<"dma_pad_3", 64, 0x68, []>;
def reg_dma_pad_4 : VPU37XX_RegisterWrapper<"dma_pad_4", 64, 0x70, []>;
def reg_dma_pad_5 : VPU37XX_RegisterWrapper<"dma_pad_5", 64, 0x78, []>;

// DMA descriptor

def reg_mapped_DMARegister : VPU37XX_RegMappedWrapper<"DMARegister",
    ["dma_link_address", "dma_watermark",
    "dma_cfg_bits",
    "dma_src",
    "dma_dst",
    "dma_length", "dma_num_planes", "dma_task_id",
    "dma_src_plane_stride", "dma_dst_plane_stride",
    "dma_attr2d_src_width", "dma_attr2d_src_stride", "dma_barriers1d_prod_mask",
    "dma_attr2d_dst_width", "dma_attr2d_dst_stride", "dma_barriers1d_cons_mask",
    "dma_barriers_prod_mask",
    "dma_barriers_cons_mask",
    "dma_barriers_sched", "dma_pad_0",
    "dma_pad_1",
    "dma_pad_2",
    "dma_pad_3",
    "dma_pad_4",
    "dma_pad_5"
    ]>;

//
// ActKernelRange registers
//

def bf_ActWLType : VPU37XX_RegFieldWrapper<"type", 8, 0>;
def reg_ActWLType : VPU37XX_RegisterWrapper<"type", 8, 0x0, ["type"]>; // WL_KERNEL = 0x00, WL_DEBUG = 0x04, WL_UNKNOWN

def bf_KernelEntry : VPU37XX_RegFieldWrapper<"kernel_entry", 32, 0>;
def reg_KernelEntry : VPU37XX_RegisterWrapper<"kernel_entry", 32, 0x4, ["kernel_entry"]>;

def bf_TextWindowBase: VPU37XX_RegFieldWrapper<"text_window_base", 32, 0>;
def reg_TextWindowBase : VPU37XX_RegisterWrapper<"text_window_base", 32, 0x8, ["text_window_base"]>;

def bf_CodeSize : VPU37XX_RegFieldWrapper<"code_size", 32, 0>;
def reg_CodeSize : VPU37XX_RegisterWrapper<"code_size", 32, 0xC, ["code_size"]>;

def bf_DataSecSize : VPU37XX_RegFieldWrapper<"data_sec_size", 32, 0>;
def reg_DataSecSize : VPU37XX_RegisterWrapper<"data_sec_size", 32, 0x10, ["data_sec_size"]>;

def bf_KernelInvoCount : VPU37XX_RegFieldWrapper<"kernel_invo_count", 32, 0>;
def reg_KernelInvoCount : VPU37XX_RegisterWrapper<"kernel_invo_count", 32, 0x14, ["kernel_invo_count"]>;

// ActKernelRange descriptor
def reg_mapped_ActKernelRangeDescriptor : VPU37XX_RegMappedWrapper<"VpuActKernelRange",
    ["type", "kernel_entry", "text_window_base", "code_size", "data_sec_size", "kernel_invo_count"]>;

//
// ActKernelInvocation registers
//

def bf_Range : VPU37XX_RegFieldWrapper<"range", 32, 0>;
def reg_Range : VPU37XX_RegisterWrapper<"range", 32, 0x0, ["range"]>;

def bf_KernelArgs : VPU37XX_RegFieldWrapper<"kernel_args", 32, 0>;
def reg_KernelArgs : VPU37XX_RegisterWrapper<"kernel_args", 32, 0x4, ["kernel_args"]>;

def bf_DataWindowBase: VPU37XX_RegFieldWrapper<"data_window_base", 32, 0>;
def reg_DataWindowBase : VPU37XX_RegisterWrapper<"data_window_base", 32, 0x8, ["data_window_base"]>;

def bf_PerfPacketOut : VPU37XX_RegFieldWrapper<"perf_packet_out", 32, 0>;
def reg_PerfPacketOut : VPU37XX_RegisterWrapper<"perf_packet_out", 32, 0xC, ["perf_packet_out"]>;

def bf_BarrierWaitMaskAct : VPU37XX_RegFieldWrapper<"barriers_wait_mask_act", 64, 0>;
def reg_BarrierWaitMaskAct : VPU37XX_RegisterWrapper<"barriers_wait_mask_act", 64, 0x10, ["barriers_wait_mask_act"]>;

def bf_BarrierWaitPostAct : VPU37XX_RegFieldWrapper<"barriers_post_mask_act", 64, 0>;
def reg_BarrierWaitPostAct : VPU37XX_RegisterWrapper<"barriers_post_mask_act", 64, 0x18, ["barriers_post_mask_act"]>;

def bf_BarrierGroupAct : VPU37XX_RegFieldWrapper<"group_act", 8, 0>;
def bf_BarrierMaskAct : VPU37XX_RegFieldWrapper<"mask_act", 8, 8>;
def reg_BarrierGroupMaskAct : VPU37XX_RegisterWrapper<"barriers_group_mask_act", 64, 0x20, ["group_act", "mask_act"]>;

def bf_ActInvoBarriersSchedStartAfter : VPU37XX_RegFieldWrapper<"act_invo_barriers_sched_start_after", 16, 0>;
def bf_ActInvoBarriersSchedCleanAfter : VPU37XX_RegFieldWrapper<"act_invo_barriers_sched_clean_after", 16, 16>;
def reg_SchedulingBarrierConfig : VPU37XX_RegisterWrapper<"act_invo_barriers_sched", 32, 0x28, ["act_invo_barriers_sched_start_after", "act_invo_barriers_sched_clean_after"]>;

def bf_InvoIndex : VPU37XX_RegFieldWrapper<"invo_index", 32, 0>;
def reg_InvoIndex : VPU37XX_RegisterWrapper<"invo_index", 32, 0x2C, ["invo_index"]>;

def bf_InvoTile : VPU37XX_RegFieldWrapper<"invo_tile", 32, 0>;
def reg_InvoTile : VPU37XX_RegisterWrapper<"invo_tile", 32, 0x30, ["invo_tile"]>;

def bf_KernelRangeIndex : VPU37XX_RegFieldWrapper<"kernel_range_index", 32, 0>;
def reg_KernelRangeIndex : VPU37XX_RegisterWrapper<"kernel_range_index", 32, 0x34, ["kernel_range_index"]>;

// ActKernelInvocation descriptor
def reg_mapped_ActKernelInvocationDescriptor : VPU37XX_RegMappedWrapper<"VpuActKernelInvocation",
    ["range", "kernel_args", "data_window_base", "perf_packet_out",
    "barriers_wait_mask_act", "barriers_post_mask_act", "barriers_group_mask_act",
    "act_invo_barriers_sched", "invo_index", "invo_tile", "kernel_range_index"]>;

//
//  BarrierCfg registers
//

def bf_NextSameId : VPU37XX_RegFieldWrapper<"next_same_id_", 16, 0, "SINT">;
def reg_NextSameId : VPU37XX_RegisterWrapper<"next_same_id_", 16, 0x0, ["next_same_id_"]>;

def bf_ProducerCount: VPU37XX_RegFieldWrapper<"producer_count_", 16, 0>;
def reg_ProducerCount : VPU37XX_RegisterWrapper<"producer_count_", 16, 0x2, ["producer_count_"]>;

def bf_ConsumerCount : VPU37XX_RegFieldWrapper<"consumer_count_", 16, 0>;
def reg_ConsumerCount : VPU37XX_RegisterWrapper<"consumer_count_", 16, 0x4, ["consumer_count_"]>;

def bf_BarrierId : VPU37XX_RegFieldWrapper<"real_id_", 8, 0>;
def reg_BarrierId : VPU37XX_RegisterWrapper<"real_id_", 16, 0x6, ["real_id_"]>;


// BarrierCfg descriptor
def reg_mapped_BarrierDescriptor : VPU37XX_RegMappedWrapper<"VpuBarrierCountConfig",
    ["next_same_id_", "producer_count_", "consumer_count_", "real_id_"]>;

#endif
