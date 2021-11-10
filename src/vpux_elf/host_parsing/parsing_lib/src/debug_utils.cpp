/*
 * {% copyright %}
 */
#include <debug_utils.h>
#include <pipePrintInit.h>
#include <nn_log.h>

#define PRINTF(...) nnLog(MVLOG_WARN, __VA_ARGS__)

namespace parsing_lib {

// Allow pipe buffer to flush properly...
static inline void delayPrint (void) {
    static tyHwPlatform platformType = static_cast<tyHwPlatform>(GET_PLATFORM_TYPE);
    if (platformType == PLATFORM_FPGA) {
       // sleep(1);
    }
}

void DebugPrintRegister(const host_parsing::DPUVariant &variant) {
    auto invariant = reinterpret_cast<host_parsing::DPUInvariant*>(variant.invariant_addr);
    PRINTF("XxXx Debugging DPU registers XxXx");
    delayPrint();
    PRINTF("NCE_DPU_SE_ADDR[0x%x]:0x%x", NCE_DPU_SE_ADDR_OFFSET, invariant->registers.se_sp_addr[0].se_addr);
    PRINTF("    -> se_addr: 0x%x", invariant->registers.se_sp_addr[0].se_addr);
    PRINTF("NCE_DPU_SPARSITY_ADDR[0x%x]:0x%x", NCE_DPU_SPARSITY_ADDR_OFFSET,
           invariant->registers.se_sp_addr[0].sparsity_addr);
    PRINTF("    -> sparsity_addr: 0x%x", invariant->registers.se_sp_addr[0].sparsity_addr);
    PRINTF("NCE_DPU_SE_ADDR1[0x%x]:0x%x", NCE_DPU_SE_ADDR1_OFFSET,
           invariant->registers.se_sp_addr[1].se_addr);
    PRINTF("    -> se_addr1: 0x%x", invariant->registers.se_sp_addr[1].se_addr);
    PRINTF("NCE_DPU_SPARSITY_ADDR1[0x%x]:0x%x", NCE_DPU_SPARSITY_ADDR1_OFFSET,
           invariant->registers.se_sp_addr[1].sparsity_addr);
    PRINTF("    -> sparsity_addr1: 0x%x", invariant->registers.se_sp_addr[1].sparsity_addr);
    PRINTF("NCE_DPU_SE_ADDR2[0x%x]:0x%x", NCE_DPU_SE_ADDR2_OFFSET,
           invariant->registers.se_sp_addr[2].se_addr);
    PRINTF("    -> se_addr2: 0x%x", invariant->registers.se_sp_addr[2].se_addr);
    PRINTF("NCE_DPU_SPARSITY_ADDR2[0x%x]:0x%x", NCE_DPU_SPARSITY_ADDR2_OFFSET,
           invariant->registers.se_sp_addr[2].sparsity_addr);
    PRINTF("    -> sparsity_addr2: 0x%x", invariant->registers.se_sp_addr[2].sparsity_addr);
    PRINTF("NCE_DPU_SE_ADDR3[0x%x]:0x%x", NCE_DPU_SE_ADDR3_OFFSET,
           invariant->registers.se_sp_addr[3].se_addr);
    PRINTF("    -> se_addr3: 0x%x", invariant->registers.se_sp_addr[3].se_addr);
    PRINTF("NCE_DPU_SPARSITY_ADDR3[0x%x]:0x%x", NCE_DPU_SPARSITY_ADDR3_OFFSET,
           invariant->registers.se_sp_addr[3].sparsity_addr);
    PRINTF("    -> sparsity_addr3: 0x%x", invariant->registers.se_sp_addr[3].sparsity_addr);
    PRINTF("NCE_DPU_SE_SP_SIZE[0x%x]:0x%x", NCE_DPU_SE_SP_SIZE_OFFSET,
           invariant->registers.se_sp_size[0].se_sp_size);
    PRINTF("    -> se_seg_size: 0x%x", invariant->registers.se_sp_size[0].se_sp_size_bf.se_seg_size);
    PRINTF("    -> sp_seg_size: 0x%x", invariant->registers.se_sp_size[0].se_sp_size_bf.sp_seg_size);
    PRINTF("NCE_DPU_SE_SP_SIZE1[0x%x]:0x%x", NCE_DPU_SE_SP_SIZE1_OFFSET,
           invariant->registers.se_sp_size[1].se_sp_size);
    PRINTF("    -> se_seg_size1: 0x%x", invariant->registers.se_sp_size[1].se_sp_size_bf.se_seg_size);
    PRINTF("    -> sp_seg_size1: 0x%x", invariant->registers.se_sp_size[1].se_sp_size_bf.sp_seg_size);
    PRINTF("NCE_DPU_SE_SP_SIZE2[0x%x]:0x%x", NCE_DPU_SE_SP_SIZE2_OFFSET,
           invariant->registers.se_sp_size[2].se_sp_size);
    PRINTF("    -> se_seg_size2: 0x%x", invariant->registers.se_sp_size[2].se_sp_size_bf.se_seg_size);
    PRINTF("    -> sp_seg_size2: 0x%x", invariant->registers.se_sp_size[2].se_sp_size_bf.sp_seg_size);
    delayPrint();
    PRINTF("NCE_DPU_Z_CONFIG[0x%x]:0x%x", NCE_DPU_Z_CONFIG_OFFSET, invariant->registers.z_config.z_config);
#ifdef CONFIG_TARGET_SOC_3720
    PRINTF("    -> addr_format_sel: 0x%x", invariant->registers.z_config.z_config_bf.addr_format_sel);
    PRINTF("    -> cm_sp_pattern: 0x%x", invariant->registers.z_config.z_config_bf.cm_sp_pattern);
    PRINTF("    -> num_ses_in_z_dir: 0x%x", invariant->registers.z_config.z_config_bf.num_ses_in_z_dir);
    PRINTF("    -> se_z_split: 0x%x", invariant->registers.z_config.z_config_bf.se_z_split);
#endif
    delayPrint();
    PRINTF("NCE_DPU_KERNAL_PAD_CFG[0x%x]:0x%x", NCE_DPU_KERNAL_PAD_CFG_OFFSET,
           invariant->registers.kernel_pad_cfg.kernel_pad_cfg);
#ifdef CONFIG_TARGET_SOC_3720
    PRINTF("    -> rst_ctxt: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.rst_ctxt);
    PRINTF("    -> sp_se_tbl_segment: 0x%x",
           invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.sp_se_tbl_segment);
    PRINTF("    -> pool_opt_en: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.pool_opt_en);
    PRINTF("    -> layer1_cmp_en: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.layer1_cmp_en);
    PRINTF("    -> layer1_wt_sp_ins: 0x%x",
           invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.layer1_wt_sp_ins);
    PRINTF("    -> dw_wt_sp_ins: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.dw_wt_sp_ins);
    PRINTF("    -> dynamic_bw_en: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.dynamic_bw_en);
    PRINTF("    -> stride_y: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.stride_y);
    PRINTF("    -> stride_y_en: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.stride_y_en);
    PRINTF("    -> wt_dense: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_dense);
    PRINTF("    -> act_dense: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.act_dense);
    PRINTF("    -> wt_plt_cfg: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.wt_plt_cfg);
    PRINTF("    -> kernel_x: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_x);
    PRINTF("    -> kernel_y: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.kernel_y);
    PRINTF("    -> pad_top_en: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.pad_top_en);
    PRINTF("    -> pad_bottom_en: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.pad_bottom_en);
    PRINTF("    -> pad_left_en: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.pad_left_en);
    PRINTF("    -> pad_right_en: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.pad_right_en);
    PRINTF("    -> mpe_assign: 0x%x", invariant->registers.kernel_pad_cfg.kernel_pad_cfg_bf.mpe_assign);
#endif

#ifdef CONFIG_TARGET_SOC_3720
    PRINTF("NCE_DPU_WEIGHT_SIZE[0x%x]:0x%x", NCE_DPU_WEIGHT_SIZE_OFFSET, variant.registers.weight_size);
    PRINTF("    -> weight_size :0x%x", variant.registers.weight_size);
#endif

#ifdef CONFIG_TARGET_SOC_3720
    PRINTF("NCE_DPU_WEIGHT_NUM[0x%x]:0x%x", NCE_DPU_WEIGHT_NUM_OFFSET, variant.registers.weight_num);
    PRINTF("    -> weight_num :0x%x", variant.registers.weight_num);
#endif

    PRINTF("NCE_DPU_WEIGHT_START[0x%x]:0x%x", NCE_DPU_WEIGHT_START_OFFSET, invariant->registers.weight_start);
    PRINTF("    -> weight_start :0x%x", invariant->registers.weight_start);
    delayPrint();
    PRINTF("NCE_DPU_TENSOR_SIZE0[0x%x]:0x%x", NCE_DPU_TENSOR_SIZE0_OFFSET,
           invariant->registers.tensor_size0.tensor_size0);
    PRINTF("    -> tensor_size_y: 0x%x", invariant->registers.tensor_size0.tensor_size0_bf.tensor_size_y);
    PRINTF("    -> tensor_size_x: 0x%x", invariant->registers.tensor_size0.tensor_size0_bf.tensor_size_x);
    PRINTF("NCE_DPU_TENSOR_SIZE1[0x%x]:0x%x", NCE_DPU_TENSOR_SIZE1_OFFSET,
           invariant->registers.tensor_size1.tensor_size1);
    PRINTF("    -> tensor_size_z: 0x%x", invariant->registers.tensor_size1.tensor_size1_bf.tensor_size_z);
    PRINTF("NCE_DPU_TENSOR_START[0x%x]:0x%x", NCE_DPU_TENSOR_START_OFFSET, invariant->registers.tensor_start);
    PRINTF("    -> tensor_start: 0x%x", invariant->registers.tensor_start);
    PRINTF("NCE_DPU_TENSOR_MODE[0x%x]:0x%x", NCE_DPU_TENSOR_MODE_OFFSET,
           invariant->registers.tensor_mode.tensor_mode);
    PRINTF("    -> pad_value: 0x%x", invariant->registers.tensor_mode.tensor_mode_bf.pad_value);
    PRINTF("    -> workload_operation: 0x%x",
           invariant->registers.tensor_mode.tensor_mode_bf.workload_operation);
    PRINTF("    -> cm_input: 0x%x", invariant->registers.tensor_mode.tensor_mode_bf.cm_input);
    PRINTF("    -> dw_input: 0x%x", invariant->registers.tensor_mode.tensor_mode_bf.dw_input);
    PRINTF("    -> zm_input: 0x%x", invariant->registers.tensor_mode.tensor_mode_bf.zm_input);
    PRINTF("    -> stride: 0x%x", invariant->registers.tensor_mode.tensor_mode_bf.stride);
    PRINTF("    -> amode: 0x%x", invariant->registers.tensor_mode.tensor_mode_bf.amode);
    PRINTF("    -> wmode: 0x%x", invariant->registers.tensor_mode.tensor_mode_bf.wmode);
    PRINTF("NCE_DPU_ELOPS_SPARSITY_ADDR[0x%x]:0x%x", NCE_DPU_ELOPS_SPARSITY_ADDR_OFFSET,
           invariant->registers.elop_sparsity_addr);
    PRINTF("    -> elop_sparsity_addr :0x%x", invariant->registers.elop_sparsity_addr);
    PRINTF("NCE_DPU_ELOPS_SE_ADDR[0x%x]:0x%x", NCE_DPU_ELOPS_SE_ADDR_OFFSET,
           invariant->registers.elop_se_addr);
    PRINTF("    -> elop_se_addr :0x%x", invariant->registers.elop_se_addr);

    delayPrint();
    PRINTF("NCE_DPU_ELOPS_WLOAD[0x%x]:0x%x", NCE_DPU_ELOPS_WLOAD_OFFSET,
           invariant->registers.elops_wload.elops_wload);
#ifdef CONFIG_TARGET_SOC_3720
    PRINTF("    -> pool_wt_rd_dis :0x%x", invariant->registers.elops_wload.elops_wload_bf.pool_wt_rd_dis);
    PRINTF("    -> pool_wt_data :0x%x", invariant->registers.elops_wload.elops_wload_bf.pool_wt_data);
    PRINTF("    -> elop_wload_type :0x%x", invariant->registers.elops_wload.elops_wload_bf.elop_wload_type);
    PRINTF("    -> fifo_wr_wload :0x%x", invariant->registers.elops_wload.elops_wload_bf.fifo_wr_wload);
    PRINTF("    -> seed_wload :0x%x", invariant->registers.elops_wload.elops_wload_bf.seed_wload);
    PRINTF("    -> elop_wload :0x%x", invariant->registers.elops_wload.elops_wload_bf.elop_wload);
#endif

    PRINTF("NCE_DPU_ACT0_OFFSET[0x%x]:0x%x", NCE_DPU_ACT0_OFFSET_OFFSET, invariant->registers.act_offset[0]);
    PRINTF("    -> adr0_offset :0x%x", invariant->registers.act_offset[0]);
    PRINTF("NCE_DPU_ACT1_OFFSET[0x%x]:0x%x", NCE_DPU_ACT1_OFFSET_OFFSET, invariant->registers.act_offset[1]);
    PRINTF("    -> adr1_offset :0x%x", invariant->registers.act_offset[1]);
    PRINTF("NCE_DPU_ACT2_OFFSET[0x%x]:0x%x", NCE_DPU_ACT2_OFFSET_OFFSET, invariant->registers.act_offset[2]);
    PRINTF("    -> adr2_offset :0x%x", invariant->registers.act_offset[2]);
    PRINTF("NCE_DPU_ACT3_OFFSET[0x%x]:0x%x", NCE_DPU_ACT3_OFFSET_OFFSET, invariant->registers.act_offset[3]);
    PRINTF("    -> adr3_offset :0x%x", invariant->registers.act_offset[3]);
    PRINTF("NCE_DPU_BASE_OFFSETA[0x%x]:0x%x", NCE_DPU_BASE_OFFSETA_OFFSET,
           invariant->registers.base_offset_a);
    PRINTF("NCE_DPU_BASE_OFFSETB[0x%x]:0x%x", NCE_DPU_BASE_OFFSETB_OFFSET,
           invariant->registers.base_offset_b);
    delayPrint();
    PRINTF("NCE_DPU_WT_OFFSET[0x%x]:0x%x", NCE_DPU_WT_OFFSET_OFFSET, invariant->registers.wt_offset);
    PRINTF("    -> wt_offset: 0x%x", invariant->registers.wt_offset);
    PRINTF("NCE_DPU_ODU_CFG[0x%x]:0x%x", NCE_DPU_ODU_CFG_OFFSET, invariant->registers.odu_cfg.odu_cfg);
#ifdef CONFIG_TARGET_SOC_3720
    PRINTF("    -> reserved_3: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.reserved_3);
    PRINTF("    -> debug_mode: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.debug_mode);
    PRINTF("    -> permutation: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.permutation);
    PRINTF("    -> nthw: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.nthw);
    PRINTF("    -> reserved_2: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.reserved_2);
    PRINTF("    -> swizzle_key: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.swizzle_key);
    PRINTF("    -> grid: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.grid);
    PRINTF("    -> mode: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.mode);
    PRINTF("    -> write_ac: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.write_ac);
    PRINTF("    -> write_pt: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.write_pt);
    PRINTF("    -> write_sp: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.write_sp);
    PRINTF("    -> reserved_1: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.reserved_1);
    PRINTF("    -> sp_out_en: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.sp_out_en);
    PRINTF("    -> sp_value: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.sp_value);
    PRINTF("    -> reserved_0: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.reserved_0);
    PRINTF("    -> dtype: 0x%x", invariant->registers.odu_cfg.odu_cfg_bf.dtype);
#endif
    delayPrint();

#ifdef CONFIG_TARGET_SOC_3720
    PRINTF("NCE_DPU_ODU_CTX_SIZE[0x%x]:0x%x", NCE_DPU_ODU_BE_SIZE_OFFSET, invariant->registers.odu_be_size);
    PRINTF("    -> odu_ctx_size:0x%x", invariant->registers.odu_be_size);
    PRINTF("NCE_DPU_ODU_CTX_THRESHOLD[0x%x]:0x%x", NCE_DPU_ODU_BE_CNT_OFFSET,
           invariant->registers.odu_be_cnt);
    PRINTF("    -> odu_ctx_cnt:0x%x", invariant->registers.odu_be_cnt);
#endif

    PRINTF("NCE_DPU_ODU_SE_SIZE[0x%x]:0x%x", NCE_DPU_ODU_SE_SIZE_OFFSET, invariant->registers.se_size);
    PRINTF("    -> se_size: 0x%x", invariant->registers.se_size);
    PRINTF("NCE_DPU_ODU_TE_DIM_0[0x%x]:0x%x", NCE_DPU_ODU_TE_DIM_0_OFFSET,
           invariant->registers.te_dim0.te_dim0);
    PRINTF("    -> te_dim_z: 0x%x", invariant->registers.te_dim0.te_dim0_bf.te_dim_z);
    PRINTF("    -> te_dim_y: 0x%x", invariant->registers.te_dim0.te_dim0_bf.te_dim_y);
    PRINTF("NCE_DPU_ODU_TE_DIM_1[0x%x]:0x%x", NCE_DPU_ODU_TE_DIM_1_OFFSET,
           invariant->registers.te_dim1.te_dim1);
    PRINTF("    -> te_dim_x: 0x%x", invariant->registers.te_dim1.te_dim1_bf.te_dim_x);
    PRINTF("NCE_DPU_ODU_PT_BASE[0x%x]:0x%x", NCE_DPU_ODU_PT_BASE_OFFSET, invariant->registers.pt_base);
    PRINTF("NCE_DPU_ODU_SP_BASE[0x%x]:0x%x", NCE_DPU_ODU_SP_BASE_OFFSET, invariant->registers.sp_base);
    PRINTF("NCE_DPU_ODU_BASE_PTR_A[0x%x]:0x%x", NCE_DPU_ODU_BASE_PTR_A_OFFSET,
           invariant->registers.base_ptr_a);
    PRINTF("NCE_DPU_ODU_BASE_PTR_B[0x%x]:0x%x", NCE_DPU_ODU_BASE_PTR_B_OFFSET,
           invariant->registers.base_ptr_b);
    leonPipePrintFlushBuffer();
    PRINTF("NCE_DPU_ODU_BASE_ADR_0[0x%x]:0x%x", NCE_DPU_ODU_BASE_ADR_0_OFFSET,
           invariant->registers.base_adr[0]);
    PRINTF("    -> base_adr_0: 0x%x", invariant->registers.base_adr[0]);
    PRINTF("NCE_DPU_ODU_BASE_ADR_1[0x%x]:0x%x", NCE_DPU_ODU_BASE_ADR_1_OFFSET,
           invariant->registers.base_adr[1]);
    PRINTF("    -> base_adr_1: 0x%x", invariant->registers.base_adr[1]);
    PRINTF("NCE_DPU_ODU_BASE_ADR_2[0x%x]:0x%x", NCE_DPU_ODU_BASE_ADR_2_OFFSET,
           invariant->registers.base_adr[2]);
    PRINTF("    -> base_adr_2: 0x%x", invariant->registers.base_adr[2]);
    PRINTF("NCE_DPU_ODU_BASE_ADR_3[0x%x]:0x%x", NCE_DPU_ODU_BASE_ADR_3_OFFSET,
           invariant->registers.base_adr[3]);
    PRINTF("    -> base_adr_3: 0x%x", invariant->registers.base_adr[3]);
    delayPrint();
    PRINTF("NCE_DPU_ODU_CAST_0[0x%x]:0x%x", NCE_DPU_ODU_CAST_0_OFFSET,
           invariant->registers.odu_cast[0].odu_cast);
    PRINTF("    -> cast_offset_0: 0x%x", invariant->registers.odu_cast[0].odu_cast_bf.cast_offset);
    PRINTF("    -> cast_enable_0: 0x%x", invariant->registers.odu_cast[0].odu_cast_bf.cast_enable);
    PRINTF("NCE_DPU_ODU_CAST_1[0x%x]:0x%x", NCE_DPU_ODU_CAST_1_OFFSET,
           invariant->registers.odu_cast[1].odu_cast);
    PRINTF("    -> cast_offset_1: 0x%x", invariant->registers.odu_cast[1].odu_cast_bf.cast_offset);
    PRINTF("    -> cast_enable_1: 0x%x", invariant->registers.odu_cast[1].odu_cast_bf.cast_enable);
    PRINTF("NCE_DPU_ODU_CAST_2[0x%x]:0x%x", NCE_DPU_ODU_CAST_2_OFFSET,
           invariant->registers.odu_cast[2].odu_cast);
    PRINTF("    -> cast_offset_2: 0x%x", invariant->registers.odu_cast[2].odu_cast_bf.cast_offset);
    PRINTF("    -> cast_enable_2: 0x%x", invariant->registers.odu_cast[2].odu_cast_bf.cast_enable);

    PRINTF("NCE_DPU_MPE_CFG[0x%x]:0x%x", NCE_DPU_MPE_CFG_OFFSET, invariant->registers.mpe_cfg.mpe_cfg);
#ifdef CONFIG_TARGET_SOC_3720
    PRINTF("    -> mpe_ftz: 0x%x", invariant->registers.mpe_cfg.mpe_cfg_bf.mpe_ftz);
    PRINTF("    -> mpe_daz: 0x%x", invariant->registers.mpe_cfg.mpe_cfg_bf.mpe_daz);
#endif
    PRINTF("    -> mrm_act_dense: 0x%x", invariant->registers.mpe_cfg.mpe_cfg_bf.mrm_act_dense);
    PRINTF("    -> mrm_weight_dense: 0x%x", invariant->registers.mpe_cfg.mpe_cfg_bf.mrm_weight_dense);
    PRINTF("    -> mpe_dense: 0x%x", invariant->registers.mpe_cfg.mpe_cfg_bf.mpe_dense);
    PRINTF("    -> mpe_mode: 0x%x", invariant->registers.mpe_cfg.mpe_cfg_bf.mpe_mode);
    PRINTF("    -> mpe_actbias: 0x%x", invariant->registers.mpe_cfg.mpe_cfg_bf.mpe_actbias);
    PRINTF("    -> mpe_wtbias: 0x%x", invariant->registers.mpe_cfg.mpe_cfg_bf.mpe_wtbias);

    delayPrint();
    PRINTF("NCE_DPU_MPE_BUS_DATA_SEL[0x%x]:0x%x", NCE_DPU_MPE_BUS_DATA_SEL_OFFSET,
           invariant->registers.mpe_bus_data_sel);

#ifdef CONFIG_TARGET_SOC_3720
    PRINTF("NCE_DPU_ELOP_SCALE[0x%x]:0x%x", NCE_DPU_ELOP_SCALE_OFFSET,
           invariant->registers.elop_scale.elop_scale);
    PRINTF("    -> elop_scale_a: 0x%x", invariant->registers.elop_scale.elop_scale_bf.elop_scale_a);
    PRINTF("    -> elop_scale_b: 0x%x", invariant->registers.elop_scale.elop_scale_bf.elop_scale_b);
#endif

    PRINTF("NCE_DPU_PPE_CFG[0x%x]:0x%x", NCE_DPU_PPE_CFG_OFFSET, invariant->registers.ppe_cfg.ppe_cfg);
    PRINTF("    -> ppe_g8_bias_a: 0x%x", invariant->registers.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_a);
    PRINTF("    -> ppe_g8_bias_b: 0x%x", invariant->registers.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_b);
    PRINTF("    -> ppe_g8_bias_c: 0x%x", invariant->registers.ppe_cfg.ppe_cfg_bf.ppe_g8_bias_c);

    PRINTF("NCE_DPU_PPE_BIAS[0x%x]:0x%x", NCE_DPU_PPE_BIAS_OFFSET, invariant->registers.ppe_bias);
    PRINTF("    -> ppe_bias: 0x%x", invariant->registers.ppe_bias);

    PRINTF("NCE_DPU_PPE_SCALE[0x%x]:0x%x", NCE_DPU_PPE_SCALE_OFFSET,
           invariant->registers.ppe_scale.ppe_scale);
    PRINTF("    -> ppe_scale_mult: 0x%x", invariant->registers.ppe_scale.ppe_scale_bf.ppe_scale_mult);
    PRINTF("    -> ppe_scale_round: 0x%x", invariant->registers.ppe_scale.ppe_scale_bf.ppe_scale_round);
    PRINTF("    -> ppe_scale_shift: 0x%x", invariant->registers.ppe_scale.ppe_scale_bf.ppe_scale_shift);

    PRINTF("NCE_DPU_PPE_SCALE_CTRL[0x%x]:0x%x", NCE_DPU_PPE_SCALE_CTRL_OFFSET,
           invariant->registers.ppe_scale_ctrl.ppe_scale_ctrl);
#ifdef CONFIG_TARGET_SOC_3720
    PRINTF("    -> ppe_fp_scale_override: 0x%x",
           invariant->registers.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_fp_scale_override);
    PRINTF("    -> ppe_scale_override: 0x%x",
           invariant->registers.ppe_scale_ctrl.ppe_scale_ctrl_bf.ppe_scale_override);
#endif
    delayPrint();

#ifdef CONFIG_TARGET_SOC_3720
    PRINTF("NCE_DPU_PPE_PRELU[0x%x]:0x%x", NCE_DPU_PPE_PRELU_OFFSET,
           invariant->registers.ppe_prelu.ppe_prelu);
    PRINTF("    -> ppe_prelu_mult: 0x%x", invariant->registers.ppe_prelu.ppe_prelu_bf.ppe_prelu_mult);
    PRINTF("    -> ppe_prelu_shift: 0x%x", invariant->registers.ppe_prelu.ppe_prelu_bf.ppe_prelu_shift);
#endif

    PRINTF("NCE_DPU_PPE_SCALE_HCLAMP[0x%x]:0x%x", NCE_DPU_PPE_SCALE_HCLAMP_OFFSET,
           invariant->registers.ppe_scale_hclamp);
    PRINTF("    -> ppe_scale_hclamp: 0x%x", invariant->registers.ppe_scale_hclamp);
    PRINTF("NCE_DPU_PPE_SCALE_LCLAMP[0x%x]:0x%x", NCE_DPU_PPE_SCALE_LCLAMP_OFFSET,
           invariant->registers.ppe_scale_lclamp);
    PRINTF("    -> ppe_scale_lclamp: 0x%x", invariant->registers.ppe_scale_lclamp);

    PRINTF("NCE_DPU_PPE_MISC[0x%x]:0x%x", NCE_DPU_PPE_MISC_OFFSET, invariant->registers.ppe_misc.ppe_misc);
#ifdef CONFIG_TARGET_SOC_3720
    PRINTF("    -> ppe_i32_convert: 0x%x", invariant->registers.ppe_misc.ppe_misc_bf.ppe_i32_convert);
    PRINTF("    -> ppe_fp16_clamp: 0x%x", invariant->registers.ppe_misc.ppe_misc_bf.ppe_fp16_clamp);
    PRINTF("    -> ppe_fp16_ftz: 0x%x", invariant->registers.ppe_misc.ppe_misc_bf.ppe_fp16_ftz);
#endif

#ifdef CONFIG_TARGET_SOC_3720
    PRINTF("NCE_DPU_PPE_FP_BIAS[0x%x]:0x%x", NCE_DPU_PPE_FP_BIAS_OFFSET, invariant->registers.ppe_fp_bias);
    PRINTF("    -> ppe_fp_bias: 0x%x", invariant->registers.ppe_fp_bias);
    PRINTF("NCE_DPU_PPE_FP_SCALE[0x%x]:0x%x", NCE_DPU_PPE_FP_SCALE_OFFSET, invariant->registers.ppe_fp_scale);
    PRINTF("    -> ppe_fp_scale: 0x%x", invariant->registers.ppe_fp_scale);
    PRINTF("NCE_DPU_PPE_FP_PRELU[0x%x]:0x%x", NCE_DPU_PPE_FP_PRELU_OFFSET, invariant->registers.ppe_fp_prelu);
    PRINTF("    -> ppe_fp_prelu: 0x%x", invariant->registers.ppe_fp_prelu);
    delayPrint();
    PRINTF("NCE_DPU_PPE_FP_CFG[0x%x]:0x%x", NCE_DPU_PPE_FP_CFG_OFFSET,
           invariant->registers.ppe_fp_cfg.ppe_fp_cfg);
    PRINTF("    -> ppe_fp_prelu_en: 0x%x", invariant->registers.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_prelu_en);
    PRINTF("    -> ppe_bf16_round: 0x%x", invariant->registers.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_bf16_round);
    PRINTF("    -> ppe_fp_bypass: 0x%x", invariant->registers.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_bypass);
    PRINTF("    -> ppe_fp_convert: 0x%x", invariant->registers.ppe_fp_cfg.ppe_fp_cfg_bf.ppe_fp_convert);
#endif

    PRINTF("NCE_DPU_WORKLOAD_SIZE0[0x%x]:0x%x", NCE_DPU_WORKLOAD_SIZE0_OFFSET,
           variant.registers.workload_size0.workload_size0);
    PRINTF("    -> workload_size_y: 0x%x", variant.registers.workload_size0.workload_size0_bf.workload_size_y);
    PRINTF("    -> workload_size_x: 0x%x", variant.registers.workload_size0.workload_size0_bf.workload_size_x);
    leonPipePrintFlushBuffer();
    PRINTF("NCE_DPU_WORKLOAD_SIZE1[0x%x]:0x%x", NCE_DPU_WORKLOAD_SIZE1_OFFSET,
           variant.registers.workload_size1.workload_size1);
    PRINTF("    -> pad_count_right: 0x%x", variant.registers.workload_size1.workload_size1_bf.pad_count_right);
    PRINTF("    -> pad_count_down: 0x%x", variant.registers.workload_size1.workload_size1_bf.pad_count_down);
    PRINTF("    -> pad_count_left: 0x%x", variant.registers.workload_size1.workload_size1_bf.pad_count_left);
    PRINTF("    -> pad_count_up: 0x%x", variant.registers.workload_size1.workload_size1_bf.pad_count_up);
    PRINTF("    -> workload_size_z: 0x%x", variant.registers.workload_size1.workload_size1_bf.workload_size_z);
    PRINTF("NCE_DPU_WORKLOAD_START0[0x%x]:0x%x", NCE_DPU_WORKLOAD_START0_OFFSET,
           variant.registers.workload_start0.workload_start0);
    PRINTF("    -> workload_start_y: 0x%x", variant.registers.workload_start0.workload_start0_bf.workload_start_y);
    PRINTF("    -> workload_start_x: 0x%x", variant.registers.workload_start0.workload_start0_bf.workload_start_x);
    PRINTF("NCE_DPU_WORKLOAD_START1[0x%x]:0x%x", NCE_DPU_WORKLOAD_START1_OFFSET,
           variant.registers.workload_start1.workload_start1);
    PRINTF("    -> workload_start_z: 0x%x", variant.registers.workload_start1.workload_start1);
    delayPrint();

#ifdef CONFIG_TARGET_SOC_3720
    PRINTF("NCE_DPU_OFFSET_ADDR[0x%x]:0x%x", NCE_DPU_OFFSET_ADDR_OFFSET, variant.registers.offset_addr.offset_addr);
    PRINTF("    -> wt_swizzle_sel: 0x%x", variant.registers.offset_addr.offset_addr_bf.wt_swizzle_sel);
    PRINTF("    -> wt_swizzle_key: 0x%x", variant.registers.offset_addr.offset_addr_bf.wt_swizzle_key);
    PRINTF("    -> idu_dbg_en: 0x%x", variant.registers.offset_addr.offset_addr_bf.idu_dbg_en);
    PRINTF("    -> swizzle_key: 0x%x", variant.registers.offset_addr.offset_addr_bf.swizzle_key);
    PRINTF("    -> idx_quad: 0x%x", variant.registers.offset_addr.offset_addr_bf.idx_quad);
    PRINTF("    -> dense_se: 0x%x", variant.registers.offset_addr.offset_addr_bf.dense_se);
    PRINTF("    -> conv_cond: 0x%x", variant.registers.offset_addr.offset_addr_bf.conv_cond);
    PRINTF("    -> bin_cfg: 0x%x", variant.registers.offset_addr.offset_addr_bf.bin_cfg);
    PRINTF("    -> nthw_ntk: 0x%x", variant.registers.offset_addr.offset_addr_bf.nthw_ntk);
#endif
    delayPrint();
    PRINTF("NCE_DPU_ODU_TE_END_0[0x%x]:0x%x", NCE_DPU_ODU_TE_END_0_OFFSET, variant.registers.te_end0.te_end0);
    PRINTF("    -> te_end_z: 0x%x", variant.registers.te_end0.te_end0_bf.te_end_z);
    PRINTF("    -> te_end_y: 0x%x", variant.registers.te_end0.te_end0_bf.te_end_y);
    PRINTF("NCE_DPU_ODU_TE_END_1[0x%x]:0x%x", NCE_DPU_ODU_TE_END_1_OFFSET, variant.registers.te_end1.te_end1);
    PRINTF("    -> te_end_x: 0x%x", variant.registers.te_end1.te_end1_bf.te_end_x);
    PRINTF("NCE_DPU_ODU_TE_BEG_0[0x%x]:0x%x", NCE_DPU_ODU_TE_BEG_0_OFFSET, variant.registers.te_beg0.te_beg0);
    PRINTF("    -> te_beg_z: 0x%x", variant.registers.te_beg0.te_beg0_bf.te_beg_z);
    PRINTF("    -> te_beg_y: 0x%x", variant.registers.te_beg0.te_beg0_bf.te_beg_y);
    PRINTF("NCE_DPU_ODU_TE_BEG_1[0x%x]:0x%x", NCE_DPU_ODU_TE_BEG_1_OFFSET, variant.registers.te_beg1.te_beg1);
    PRINTF("    -> te_beg_x: 0x%x", variant.registers.te_beg1.te_beg1_bf.te_beg_x);
    leonPipePrintFlushBuffer();
}
} // namespace parsing_lib
