#ifndef DPU2P7_DESCRIPTOR 
#define DPU2P7_DESCRIPTOR
/* Auto-generated from RTL RGI */
#include <stdint.h>

typedef struct {
  uint32_t workload_start_odu : 1; // 0:0 "Initiates the start of an ODU workload"
  uint32_t workload_start_idu : 1; // 1:1 "Initiates the start of an IDU workload"
  uint32_t workload_prm_sel : 1; // 2:2 "Indicates whether the DPU should use the primary or secondary workload descriptor. *** UNUSED ***"
  uint32_t workload_valid : 1; // 3:3 "Indicates the current workload descriptors are valid"
  uint32_t workload_shad_odu : 1; // 4:4 "Initiates the start of an ODU workload"
  uint32_t workload_shad_idu : 1; // 5:5 "Initiates the start of an IDU workload"
  uint32_t workload_idu_auto_upd_0 : 1; // 6:6 ""10" - hardware waits for the IDU only to indicate it is IDLE before copying the IDU/MRM/MPE workload descriptor from the primary to the shadow register and starting the IDU\n"11" - hardware waits for the IDU and MPE to be in the IDLE state before copying the IDU/MRM/MPE workload descriptor from the primary regsiter to the shadow register and starting the IDU\n"01" - hardware waits for the IDU, MPE and ODU to be in the IDLE state before copying the IDU/MRM/MPE workload descriptor from the primary regsiter to the shadow register and starting the IDU\n"00" - software is responsible for copying the workload descriptor from the primary to the shadow register and starting the IDU based on status reported in DPU_STS"
  uint32_t workload_idu_auto_upd_1 : 1; // 7:7 "Refer to workload_idu_auto_upd_0 bit description"
  uint32_t workload_odu_auto_upd : 1; // 8:8 ""1" - hardware waits for the ODU to indicate it is IDLE before copying the ODU/PPE workload descriptor from the primary to the shadow register and starting the IDU\n"0" - software is responsible for copying the workload descriptor from the primary to the shadow register and starting the ODU based on status reported in DPU_STS"
  uint32_t Reserved_0 : 1; // 9:9 "Reserved"
  uint32_t Reserved_1 : 1; // 10:10 "Reserved"
  uint32_t Reserved_2 : 1; // 11:11 "Reserved"
  uint32_t rst_ctxt_new : 1; // 12:12 "When the workload type is being changed, RST_CTXT must be set on the first workload of the new type. Note that the DPU pipeline must be empty before a workload is generated where RST_CTXT is set."
  uint32_t Reserved_3 : 1; // 13:13 "Reserved"
  uint32_t Reserved_4 : 1; // 14:14 "Reserved"
  uint32_t odu_stat_clr : 1; // 15:15 "When written to 1 all ODU stats are cleared; CDC transfer in progress while this bit is 1"
  uint32_t idu_stat_clr : 1; // 16:16 "When written to 1 all IDU stats are cleared"
  uint32_t Reserved_5 : 1; // 17:17 "Reserved"
  uint32_t odu_fifo_flush : 1; // 18:18 "Flushes ODU input and output FIFOs; CDC transfer in progress while this bit is 1"
  uint32_t odu_debug_step : 1; // 19:19 "ODU debug step; CDC transfer in progress while this bit is 1"
  uint32_t odu_debug_read : 1; // 20:20 "ODU debug read; CDC transfer in progress while this bit is 1"
  uint32_t Reserved_6 : 11; // 21:31 "Reserved"
}dpu_cfg_reg;

typedef struct {
  uint32_t workload_status_odu : 1; // 0:0 "Indicates the ODU has completed the current workload"
  uint32_t workload_status_idu : 1; // 1:1 "Indicates the IDU has completed the current workload"
  uint32_t workload_status_mpe : 1; // 2:2 "Indicates that all the MPEs have completed the current workload"
  uint32_t workload_idu_auto_sts : 1; // 3:3 "Indicates the hardware has copied the IDU/MRM/MPE workload descriptor from the primary register to the shadow register and the primary register is ready to be reloaded"
  uint32_t workload_odu_auto_sts : 1; // 4:4 "Indicates the hardware has copied the ODU/PPE workload descriptor from the primary register to the shadow register and the primary register is ready to be reloaded"
}dpu_sts_reg;

typedef struct {
  uint32_t vpu2p0_rsvd_0 : 1; // 0:0 "Not used"
}vpu2p0_rsvd_0_reg;

typedef struct {
  uint32_t x : 14; // 0:13 "Workload X size - in units of activations"
  uint32_t y : 14; // 14:27 "Workload Y size - in units of activations"
}workload_size0_reg;

typedef struct {
  uint32_t z : 14; // 0:13 "Workload Z size - in units of activations [UNUSED - WEIGHT_NUM and WEIGHT START control the section of the output tensor to be processed by the IDU]"
  uint32_t pad_count_up : 3; // 14:16 "The amount to pad at the upper boundary"
  uint32_t pad_count_left : 3; // 17:19 "The amount to pad at the left boundary"
  uint32_t pad_count_down : 3; // 20:22 "The amount to pad at the bottom boundary"
  uint32_t pad_count_right : 3; // 23:25 "The amount to pad at the right boundary"
}workload_size1_reg;

typedef struct {
  uint32_t x : 14; // 0:13 "Workload X start offset - X index into the tensor to the start of the workload"
  uint32_t y : 14; // 14:27 "Workload Y start offset - Y index into the tensor to the start of the workload"
}workload_start0_reg;

typedef struct {
  uint32_t z : 14; // 0:13 "Workload Z start offset - Z index into the tensor to the start of the workload [UNUSED - WEIGHT_NUM and WEIGHT START control the section of the output tensor to be processed by the IDU]"
}workload_start1_reg;

typedef struct {
  uint32_t nthw_ntk : 2; // 0:1 "00 - NTHW_NTK_8_8\n01 - NTHW_NTK_4_16\n11 - NTHW_NTK_16_4"
  uint32_t bin_cfg : 1; // 2:2 "Binary Config"
  uint32_t conv_cond : 1; // 3:3 "Convolution continue"
  uint32_t dense_se : 1; // 4:4 "Dense SE mode"
  uint32_t idx_quad : 1; // 5:5 "Idx Quad"
  uint32_t swizzle_key : 3; // 6:8 "Swizzle key:\n 0: No swizzling\n 1-5: Usage of 1-5 stagger bits"
  uint32_t idu_mrm_clk_en : 1; // 9:9 "0 - IDU/MRM hardware clock gating is enabled - hardware enables the IDU/MRM clock as required based on IDU/MRM status\n1 - IDU/MRM hardware clock gating is disabled - IDU/MRM clock is always enabled"
  uint32_t odu_clk_en : 1; // 10:10 "0 - ODU hardware clock gating is enabled - hardware enables the ODU clock as required based on ODU status\n1 - ODU hardware clock gating is disabled - ODU clock is always enabled"
  uint32_t mpe_clk_en : 1; // 11:11 "0 - MPE hardware clock gating is enabled - hardware enables the MPE clock as required based on MPE status\n1 - MPE hardware clock gating is disabled - MPE clock is always enabled"
  uint32_t ppe_clk_en : 1; // 12:12 "0 - PPE hardware clock gating is enabled - hardware enables the PPE clock as required based on PPE status\n1 - PPE hardware clock gating is disabled - PPE clock is always enabled"
  uint32_t odu_stat_en : 1; // 13:13 "Enables gathering of ODU statistics"
  uint32_t idu_stat_en : 1; // 14:14 "Enables gathering of IDU statistics"
  uint32_t reserved_1 : 1; // 15:15 "RESERVED"
  uint32_t odu_stat_clr_mode : 1; // 16:16 "Selects mode for clearing statistics\n0 - HW auto clears the statistics at the start of every workload\n1 - SW clears statistics via the ODU_STAT_CLR register bits"
  uint32_t idu_stat_clr_mode : 1; // 17:17 "Selects mode for clearing statistics\n0 - HW auto clears the statistics at the start of every workload\n1 - SW clears statistics via the IDU_STAT_CLR register bits"
  uint32_t reserved_2 : 1; // 18:18 "RESERVED"
  uint32_t shave_l2_cache_en : 1; // 19:19 "L2 cache enable\n0 - Shave cache port is shared with ODU\n1 - Shave cache port uses the L2 cache port (default)"
  uint32_t idu_dbg_en : 2; // 20:21 "Debug mode for IDU"
  uint32_t reserved_3 : 5; // 22:26 "RESERVED"
  uint32_t wt_swizzle_key : 3; // 27:29 "By default weight requests use 'swizzle_key'\nWhen 'wt_swizzle_sel' is set, weight requests will use 'wt_swizzle_key':\n 0: No swizzling\n 1-5: Usage of 1-5 stagger bits"
  uint32_t wt_swizzle_sel : 1; // 30:30 "Select which swizzle key to use for weight reads (IDU)\n0 - Use swizzle_key (bits 8:6)\n1 - Use wt_swizzle_key"
  uint32_t reserved_4 : 1; // 31:31 "RESERVED"
}compute_stencil_reg;

typedef struct {
  uint32_t y : 13; // 0:12 "Y end coordinate of subtensor"
  uint32_t z : 13; // 13:25 "Z end coordinate of subtensor"
}te_end0_reg;

typedef struct {
  uint32_t x : 13; // 0:12 "X end coordinate of subtensor"
}te_end1_reg;

typedef struct {
  uint32_t y : 13; // 0:12 "Y start coordinate of subtensor"
  uint32_t z : 13; // 13:25 "Z start coordinate of subtensor"
}te_beg0_reg;

typedef struct {
  uint32_t x : 13; // 0:12 "X start cooridnate of subtensor"
}te_beg1_reg;

typedef struct {
  uint32_t cmx_slice0_low_addr : 28; // 0:27 "CMX slice 0 element base pointer[31:4] and last four bits tied to 4'h00"
}cmx_slice0_low_addr_reg;

typedef struct {
  uint32_t cmx_slice1_low_addr : 28; // 0:27 "CMX slice 1 element base pointer[31:4] and last four bits tied to 4'h00"
}cmx_slice1_low_addr_reg;

typedef struct {
  uint32_t cmx_slice2_low_addr : 28; // 0:27 "CMX slice 2 element base pointer.(not supported on 2.7)"
}cmx_slice2_low_addr_reg;

typedef struct {
  uint32_t cmx_slice3_low_addr : 28; // 0:27 "CMX slice 3 element base pointer.(not supported on 2.7)"
}cmx_slice3_low_addr_reg;

typedef struct {
  uint32_t cmx_slice_size : 20; // 0:19 "CMX slice size[*:4]. last four bits tied to 4'h0 so upper needs to be tied"
}cmx_slice_size_reg;

typedef struct {
  uint32_t cmx_slice0_upper_addr : 28; // 0:27 "CMX slice 0 upper pointer and any address below represents CMX slice 0"
}cmx_slice0_high_addr_reg;

typedef struct {
  uint32_t cmx_slice1_upper_addr : 28; // 0:27 "CMX slice 1 upper pointer and any address below represents CMX slice 1"
}cmx_slice1_high_addr_reg;

typedef struct {
  uint32_t cmx_slice2_upper_addr : 28; // 0:27 "CMX slice 2 upper pointer and any address below represents CMX slice 2(not supported on 2.7)"
}cmx_slice2_high_addr_reg;

typedef struct {
  uint32_t cmx_slice3_upper_addr : 28; // 0:27 "CMX slice 3 upper pointer and any address below represents CMX slice 3(not supported on 2.7)"
}cmx_slice3_high_addr_reg;

typedef struct {
  uint32_t cmx_lower_rd : 1; // 0:0 "cmx slice lower read en status i.e. outside cmx memory"
  uint32_t cmx_higher_rd : 1; // 1:1 "cmx slice higher read en status i.e. outside cmx memory"
  uint32_t unused0 : 6;
  uint32_t cmx_slice_0_rd : 1; // 8:8 "cmx slice 0 read enable status"
  uint32_t cmx_slice_1_rd : 1; // 9:9 "cmx slice 1 read enable status"
  uint32_t cmx_slice_2_rd : 1; // 10:10 "cmx slice 2 read enable status(not supported on 2.7)"
  uint32_t cmx_slice_3_rd : 1; // 11:11 "cmx slice 3 read enable status(not supported on 2.7)"
  uint32_t unused1 : 4;
  uint32_t cmx_lower_wr : 1; // 16:16 "cmx slice lower write en status i.e. outside cmx memory"
  uint32_t cmx_higher_wr : 1; // 17:17 "cmx slice higher write en status i.e. outside cmx memory"
  uint32_t unused2 : 6;
  uint32_t cmx_slice_0_wr : 1; // 24:24 "cmx slice 0 write enable status"
  uint32_t cmx_slice_1_wr : 1; // 25:25 "cmx slice 1 write enable status"
  uint32_t cmx_slice_2_wr : 1; // 26:26 "cmx slice 2 write enable status(not supported on 2.7)"
  uint32_t cmx_slice_3_wr : 1; // 27:27 "cmx slice 3 write enable status(not supported on 2.7)"
}cmx_slice_enable_status_reg;

typedef struct {
  uint32_t se_addr; // 0:31 "Storage element table base pointer (Bits [3:0] are unused). This must be set to 0 in dense mode."
}se_addr_reg;

typedef struct {
  uint32_t sparsity_addr; // 0:31 "Sparsity table base address (Bits [3:0] are unused). This must be set to 0 in dense mode."
}sparsity_addr_reg;

typedef struct {
  uint32_t se_addr1; // 0:31 "In sparse mode (i.e. when KERNAL_PAD_CFG.ACT_DENSE) contains the storage element table segment 1 pointer \n\nIn dense mode (i.e. when KERNAL_PAD_CFG.ACT_DENSE = 1) contains the pointer to the segment 1 of the dense tensor, where bits [19:0] correspond to bits [23:4] of the segment 1 start address of the dense tensor. \n\nIn depth wise mode (i.e. when TENSOR_MODE.dw_input=1) contains the pointer to the segment 1 of the dense tensor\n\n (Bits [3:0] are unused)"
}se_addr1_reg;

typedef struct {
  uint32_t sparsity_addr1; // 0:31 "Sparsity table segment 1 pointer\n\n (Bits [3:0] are unused)"
}sparsity_addr1_reg;

typedef struct {
  uint32_t se_addr2; // 0:31 "In sparse mode (i.e. when KERNAL_PAD_CFG.ACT_DENSE) contains the storage element table segment 2 pointer \n\nIn dense mode (i.e. when KERNAL_PAD_CFG.ACT_DENSE = 1) contains the pointer to the segment 2 of the dense tensor, where bits [19:0] correspond to bits [23:4] of the segment 2 start address of the dense tensor. \n\nIn depth wise mode (i.e. when TENSOR_MODE.dw_input=1) contains the pointer to the segment 2 of the dense tensor\n\n (Bits [3:0] are unused)"
}se_addr2_reg;

typedef struct {
  uint32_t sparsity_addr2; // 0:31 "Sparsity table segment 2 pointer\n\n (Bits [3:0] are unused)"
}sparsity_addr2_reg;

typedef struct {
  uint32_t se_addr3; // 0:31 "In sparse mode (i.e. when KERNAL_PAD_CFG.ACT_DENSE) contains the storage element table segment 3 pointer \n\nIn dense mode (i.e. when KERNAL_PAD_CFG.ACT_DENSE = 1) contains the pointer to the segment 3 of the dense tensor, where bits [19:0] correspond to bits [23:4] of the segment 3 start address of the dense tensor. \n\nIn depth wise mode (i.e. when TENSOR_MODE.dw_input=1) contains the pointer to the segment 3 of the dense tensor\n\n (Bits [3:0] are unused)"
}se_addr3_reg;

typedef struct {
  uint32_t sparsity_addr3; // 0:31 "Sparsity table segment 3 pointer\n\n (Bits [3:0] are unused)"
}sparsity_addr3_reg;

typedef struct {
  uint32_t sp_seg_size : 14; // 0:13 "Size of segment 0 of the sparsity table. Each unit corresponds to 128 bits of sparsity (i.e.16 bytes in CMX)."
  uint32_t se_seg_size : 18; // 14:31 "In sparse mode (i.e. when KERNAL_PAD_CFG.ACT_DENSE) contains the size of segment 0 of the storage element table. Each unit corresponds to 1 storage entry (i.e. 4 bytes in CMX). In sparse mode, each segment of the SE table must contain a multiple of 4 pointers. Therefore SE_SEG_SIZE must be configured to be a multiple of 4.\n\nIn dense mode (i.e. when KERNAL_PAD_CFG.ACT_DENSE = 1) contains the size of segment 0 of the dense tensor. Each unit corresponds to four Z columns of input tensor activations.\n\nIn depth wise mode (i.e. when TENSOR_MODE.dw_input=1) contains the size of segment 0 of the dense tensor. Each unit corresponds to one Z column of input tensor activations (i.e. 16 Activation bytes)."
}se_sp_size_reg;

typedef struct {
  uint32_t sp_seg_size1 : 14; // 0:13 "Size of segment 1 of the sparsity table. Each unit corresponds to 128 bits of sparsity (i.e.16 bytes in CMX)."
  uint32_t se_seg_size1 : 18; // 14:31 "In sparse mode (i.e. when KERNAL_PAD_CFG.ACT_DENSE) contains the size of segment 1 of the storage element table. Each unit corresponds to 1 storage entry (i.e. 4 bytes in CMX). In sparse mode, each segment of the SE table must contain a multiple of 4 pointers. Therefore SE_SEG_SIZE1 must be configured to be a multiple of 4.\n\nIn dense mode (i.e. when KERNAL_PAD_CFG.ACT_DENSE = 1) contains the size of segment 1 of the dense tensor. Each unit corresponds to four Z columns of input tensor activations.\n\nIn depth wise mode (i.e. when TENSOR_MODE.dw_input=1) contains the size of segment 1 of the dense tensor. Each unit corresponds to one Z column of input tensor activations (i.e. 16 Activation bytes)."
}se_sp_size1_reg;

typedef struct {
  uint32_t sp_seg_size2 : 14; // 0:13 "Size of segment 2 of the sparsity table. Each unit corresponds to 128 bits of sparsity (i.e.16 bytes in CMX)."
  uint32_t se_seg_size2 : 18; // 14:31 "In sparse mode (i.e. when KERNAL_PAD_CFG.ACT_DENSE) contains the size of segment 2 of the storage element table. Each unit corresponds to 1 storage entry (i.e. 4 bytes in CMX). In sparse mode, each segment of the SE table must contain a multiple of 4 pointers. Therefore SE_SEG_SIZE2 must be configured to be a multiple of 4.\n\nIn dense mode (i.e. when KERNAL_PAD_CFG.ACT_DENSE = 1) contains the size of segment 2 of the dense tensor. Each unit corresponds to four Z columns of input tensor activations.\n\nIn depth wise mode (i.e. when TENSOR_MODE.dw_input=1) contains the size of segment 2 of the dense tensor. Each unit corresponds to one Z column of input tensor activations (i.e. 16 Activation bytes)."
}se_sp_size2_reg;

typedef struct {
  uint32_t se_z_split : 4; // 0:3 "Storage element size - supported sizes are:\n 1x1x16 [SE_Z_SPLIT001]\n 1x1x32 [SE_Z_SPLIT010]\n 1x1x64 [SE_Z_SPLIT011]\n 1x1x128 [SE_Z_SPLIT100]\n 1x1x256 [SE_Z_SPLIT101]\n 1x1x512 [SE_Z_SPLIT110]\n 1x1x1024 [SE_Z_SPLIT111]\n 1x1x2048 [SE_Z_SPLIT = 1000]\n 1x1x4096 [SE_Z_SPLIT = 1001]\n 1x1x8192 [SE_Z_SPLIT000]"
  uint32_t num_ses_in_z_dir : 9; // 4:12 "Number of Storage Elements in the Z direction of input tensor \n 0 = 1 SE per Z dierction\n 1 = 2 SE's per Z direction (where the SE size is spoecified by SE_Z_SPLIT)\n … \n 511 = 512 SE's per Z direction (where the SE size is spoecified by SE_Z_SPLIT)"
  uint32_t cm_sp_pattern : 16; // 13:28 "Input layer sparsity pattern - this is used when KERNAL_PAD_CFG.LAYER1_WT_SP_INS=1"
  uint32_t unused0 : 2;
  uint32_t addr_format_sel : 1; // 31:31 "NOT USED BY RTL. USED BY MODEL TO SUPPORT LEGACY BEHAVIOUR"
}z_config_reg;

typedef struct {
  uint32_t mpe_assign : 1; // 0:0 "MPE assignment \n 0 - 4x4 \n 1 - 16x1\n[Note that for FP16 only a 4x1 grid is supported]"
  uint32_t pad_right_en : 1; // 1:1 "Pad right enable - NOTE - this bit will be redundant once the pad_count* registers are used"
  uint32_t pad_left_en : 1; // 2:2 "Pad left enable - NOTE - this bit will be redundant once the pad_count* registers are used"
  uint32_t pad_bottom_en : 1; // 3:3 "Pad bottom enable - NOTE - this bit will be redundant once the pad_count* registers are used"
  uint32_t pad_top_en : 1; // 4:4 "Pad top enable - NOTE - this bit will be redundant once the pad_count* registers are used"
  uint32_t kernel_y : 4; // 5:8 "IDU Y kernel size - supported range of 1-11\n 0001 - kernel size = 1\n 0010 - kernel size = 2 … etc"
  uint32_t kernel_x : 4; // 9:12 "IDU X kernel size - supported range of 1-11\n 0001 - kernel size = 1\n 0010 - kernel size = 2 … etc"
  uint32_t wt_plt_cfg : 2; // 13:14 "00 - no palletisation\n01 - 1 bit palletisation on weights\n10 - 2 bit palletisation on weights\n11 - 4 bit palletisation on weights"
  uint32_t act_dense : 1; // 15:15 "1 - activations are dense\n0 - activations are sparse"
  uint32_t wt_dense : 1; // 16:16 "1 - weights are dense\n0 - weights are sparse"
  uint32_t stride_y_en : 1; // 17:17 "0 - Y stride is confgured via the TENSOR_MODE.STRIDE register bits\n1 - Y stride is configured via the KERNAL_PAD_CFG.STRIDE_Y register bits"
  uint32_t stride_y : 3; // 18:20 "Y stride - STRIDE_Y_EN must be set to 1 for this to take effect."
  uint32_t dynamic_bw_en : 1; // 21:21 "When clear, each activation reader and weight reader has its own dedciated CMX port. When set invokes the following behavior:\n- When NTHW_NTK = 16_4 enables activation data requests to go out on the corresponding weight reader port\n- When NTHW_NTK = 4_16 enables weight data requests to go out on the corresponding actvation reader port\n- When NTHW_NTK = 8_8 has no impact"
  uint32_t dw_wt_sp_ins : 1; // 22:22 "When set enables the IDU to insert the required weight sparsity pattern for DW workloads. This bit must only be set when the data type is FP16, I8 or U8 and WORKLOAD_SIZE_Z = 16, 32 or 64. For all other datatypes and workload sizes the CMX must be populated with the required weight sparsity pattern and the weight configuration must be set to sparse."
  uint32_t layer1_wt_sp_ins : 1; // 23:23 "When set enables the IDU to insert the required weight sparsity pattern for the input layer. The sparity pattern is taken from the Z_CONFIG.CM_SP_PATTERN register."
  uint32_t layer1_cmp_en : 1; // 24:24 "When set the IDU reads the input layer in a compressed format where 4 pixels are stored per 16B CMX entry. When clear the IDU reads in the input layer in a non compressed format where 1 pixel is stored per 16B CMX entry."
  uint32_t pool_opt_en : 1; // 25:25 "When set enables a performance optimisation for pooling layers. This should only be used for workloads where KERNEL_SIZE=WORKLOAD_SIZE and PADDING is disabled."
  uint32_t unused0 : 3;
  uint32_t sp_se_tbl_segment : 1; // 29:29 "In sparse mode (i.e. when KERNAL_PAD_CFG.ACT_DENSE) this bit enables segmenting of the sparsity and storage element tables into four segments. Partitioning of the sparsity and storage element tables is done to reduce interslice bandwidth in the CMX.\n\n1 - The sparsity and storage element tables are partitioned into four segments - following registers must be configured to point to each segment of the sparsity and storage element table.\n - SE_ADDR, SE_ADDR1, SE_ADDR2, SE_ADDR3\n - SPARSITY_ADDR, SPARSITY_ADDR1, SPARSITY_ADDR2, SPARSITY_ADDR3\n - SE_SP_SIZE, SE_SP_SIZE1, SE_SP_SIZE2\n0 - The sparsity and storage element tables are not partitioned - only a single pointer must be configured for each table via the SE_ADDR & SPARSITY_ADDR registers.\n\nIn dense mode (i.e. when KERNAL_PAD_CFG.ACT_DENSE = 1) this bit enables segementing of the dense tensor into four segments. Partitioning of the tensors is done to reduce interslice bandwdith in the CMX.\n\n1 - The dense tensor is partitioned into 4 segments - following registers must be configured to point to each segment of the tensor:\n - TENSOR_START, SE_ADDR1, SE_ADDR2, SE_ADDR3\n - SE_SP_SIZE, SE_SP_SIZE1, SE_SP_SIZE2\n0 - The dense tensor is is not partitioned - only a single pointer must be configured for the tensor via the TENSOR_START register\n\nFor depth wise mode (i.e. when TENSOR_MODE.dw_input=1) the register behaves as for Dense mode described above."
  uint32_t rst_ctxt : 1; // 30:30 "When the workload type is being changed, RST_CTXT must be set on the first workload of the new type. Note that the DPU pipeline must be empty before a workload is generated where RST_CTXT is set."
}kernel_pad_cfg_reg;

typedef struct {
  uint32_t size : 21; // 0:20 "Weight set size (Input Z * kernel X * kernel Y)"
}weight_size_reg;

typedef struct {
  uint32_t sets : 14; // 0:13 "Number of weight sets (i.e. Z depth of output tensor). Must be a multiple of 16."
}weight_num_reg;

typedef struct {
  uint32_t weight_start; // 0:31 "Address of list of weight sparsity and weight data pointers. Each weight set has a weight sparsity and weight data pointer. The list of pointers is contiguous in memory. The number of entries in the list is specified by WEIGHT_NUM."
}weight_start_reg;

typedef struct {
  uint32_t x : 14; // 0:13 "Input Tensor X dimension - in units of activations"
  uint32_t y : 14; // 14:27 "Input Tensor Y dimension - in units of activations [UNUSED]"
}tensor_size0_reg;

typedef struct {
  uint32_t z : 14; // 0:13 "Input Tensor Z dimension - in units of activations"
}tensor_size1_reg;

typedef struct {
  uint32_t tensor_start; // 0:31 "Tensor 0,0,0 address. \n\nIn Z-Major dense mode, bits [19:0] correspond to bits [23:4] of the tensor start address. ADR0_OFFSET[31:0] can be used to add a further offset to the start of a tensor."
}tensor_start_reg;

typedef struct {
  uint32_t wmode : 4; // 0:3 "Weight data type\n 0000 - FP16\n 0001 - U8\n 0010 - I8\n 0011 - I4\n 0100 - I2\n 0101 - BF16\n 0110 - U4\n 0111 - BIN\n 1000 - FP8"
  uint32_t amode : 4; // 4:7 "Activation data type\n 0000 - FP16\n 0001 - U8\n 0010 - I8\n 0011 - I4\n 0100 - I2\n 0101 - BF16\n 0110 - U4\n 0111 - BIN\n 1000 - FP8"
  uint32_t stride : 3; // 8:10 "Stride - same stride is used in both X and Y directions. Stride =0 => stride of 1, stride = 1 => stride of 2 etc."
  uint32_t zm_input : 1; // 11:11 "When set indicates that the input input workload is to be processed as Z-major convolution"
  uint32_t dw_input : 1; // 12:12 "When set indicates that the input input workload is to be processed as depthwise convolution"
  uint32_t cm_input : 1; // 13:13 "When set indicates that the input input workload is to be processed as channel major"
  uint32_t workload_operation : 2; // 14:15 "workload operation:\n2'b00: Conv\n2'b10: Max Pool"
  uint32_t pad_value : 16; // 16:31 "Sets the value to be used when padding in DW and CM modes of operation. Prevents applied bias from making the Padded values non-zero in the MPE. Also allows an arbitrary value to be set for the padding to prevent problems during max pooling."
}tensor_mode_reg;

typedef struct {
  uint32_t elop_sparsity_addr; // 0:31 "Sparsity address base address of second input tensor for elementwise operations"
}elops_sparsity_addr_reg;

typedef struct {
  uint32_t elop_se_addr; // 0:31 "Storage element base address pointer of second input tensor for elementwise operations"
}elops_se_addr_reg;

typedef struct {
  uint32_t elop_wload : 1; // 0:0 "Elementwise workload - when set indicates that the workload is an elementwise workload - this enables the IDU to read in 2 tensors instead of a tensor and weight sets for a standard convolution."
  uint32_t seed_wload : 1; // 1:1 "Accumulator seed workload - when set indicates the the workload is to seed the MPE accumulators. The contents to seed the MPE accumulators are read from a table in memory using TENSOR_START as the pointer to the table."
  uint32_t fifo_wr_wload : 1; // 2:2 "MPE FIFO Write workload - when set indicates the the workload is to write the MPE FIFOs. The contents to be written to the MPE FIFOs are read from a table in memory using TENSOR_START as the pointer to the table."
  uint32_t elop_wload_type : 1; // 3:3 "Elementwise workload type (only valid when elop_wload=1'b1)\n1'b0 - Eltwise ADD of two tensors\n1'b1 - Single tensor operation"
  uint32_t pool_wt_data : 16; // 4:19 "When KERNAL_PAD_CFG.POOL_WT_RD_DIS is set POOL_WT_DATA specifies the data to be used in the pooling operation - this is required for average pooling where the data is expected to be set to 1."
  uint32_t unused0 : 6;
  uint32_t pool_wt_rd_dis : 1; // 26:26 "For pooling operations weight pointer and weight reading should be disabled by setting POOL_WT_RD_DIS to 1. For all other operations POOL_WT_RD_DIS must be set to 0. Note that for average pooling, weights are used but do not need to be read from CMX as they are all equal to 1. This constant can be re-configured via POOL_WT_DATA if required."
}elops_wload_reg;

typedef struct {
  uint32_t adr_offset; // 0:31 "Address offset 0. When the base offset read by the IDU matches BASE_OFFSETA.BASE_OFFSET0 then ADR0_OFFSET is added to the storage element read by the IDU and the result is the activation read address. ADR0_OFFSET[3:0] are unused."
}act0_offset_reg;

typedef struct {
  uint32_t adr_offset; // 0:31 "Address offset 1. When the base offset read by the IDU matches BASE_OFFSETA.BASE_OFFSET1 then ADR1_OFFSET is added to the storage element read by the IDU and the result is the activation read address. ADR1_OFFSET[3:0] are unused."
}act1_offset_reg;

typedef struct {
  uint32_t adr_offset; // 0:31 "Address offset 2. When the base offset read by the IDU matches BASE_OFFSETB.BASE_OFFSET0 then ADR2_OFFSET is added to the storage element read by the IDU and the result is the activation read address. ADR2_OFFSET[3:0] are unused."
}act2_offset_reg;

typedef struct {
  uint32_t adr_offset; // 0:31 "Address offset 3. When the base offset read by the IDU matches BASE_OFFSETB.BASE_OFFSET2 then ADR3_OFFSET is added to the storage element read by the IDU and the result is the activation read address. ADR3_OFFSET[3:0] are unused."
}act3_offset_reg;

typedef struct {
  uint32_t base_offset0 : 9; // 0:8 "BASE_OFFSET0 - see ACT0_OFFSET register for functional description"
  uint32_t base_offset1 : 9; // 9:17 "BASE_OFFSET1 - see ACT1_OFFSET register for functional description"
}base_offseta_reg;

typedef struct {
  uint32_t base_offset0 : 9; // 0:8 "BASE_OFFSET2 - see ACT2_OFFSET register for functional description"
  uint32_t base_offset1 : 9; // 9:17 "BASE_OFFSET3 - see ACT3_OFFSET register for functional description"
}base_offsetb_reg;

typedef struct {
  uint32_t wt_offset; // 0:31 "Weight offset. This is added to the weight sparsity and weight data address read by the IDU and allows an offest to be specified from the addresses in the weight control pointer.\nWT_OFFSET[3:0] are unused."
}wt_offset_reg;

typedef struct {
  uint32_t dtype : 4; // 0:3 "ODU output data type:\n 0: FP16\n 1: U8F\n 2: G8\n 3: I8\n 4: I32\n 5: I4\n 6: I2\n 7: LOG\n 8: BIN"
  uint32_t reserved_0 : 1; // 4:4 "Reserved"
  uint32_t sp_value : 8; // 5:12 "The value used by the ODU to determine sparsity"
  uint32_t sp_out_en : 1; // 13:13 "Output data compression:\n 0: Compression disabled\n 1: Compression enabled"
  uint32_t reserved_1 : 1; // 14:14 "Reserved"
  uint32_t write_sp : 1; // 15:15 "Sparsity enable:\n 0: No sparsity is being written out\n 1: Sparsity is being written out"
  uint32_t write_pt : 1; // 16:16 "Pointer enable:\n 0: No pointer are being written out\n 1: Pointers are being written out"
  uint32_t write_ac : 1; // 17:17 "Activation enable:\n 0: No activations are being written out\n 1: Activations are being written out"
  uint32_t mode : 2; // 18:19 "Mode:\n 0: Dense\n 1: Superdense\n 2: Sparse"
  uint32_t grid : 1; // 20:20 "Grid Configuration:\n 0: 4x4\n 1: 16x1"
  uint32_t swizzle_key : 3; // 21:23 "Swizzle key:\n 0: No swizzling\n 1-5: Usage of 1-5 stagger bits"
  uint32_t reserved_2 : 1; // 24:24 "Reserved"
  uint32_t nthw : 2; // 25:26 "NTHW Configuration:\n 0: NTHW1\n 1: NTHW4\n 2: NTHW8\n 3: NTHW16"
  uint32_t permutation : 3; // 27:29 "Permutation:\n 0: ZXY\n 1: ZYX\n 2: YZX\n 3: YXZ\n 4: XZY\n 5: XYZ"
  uint32_t debug_mode : 1; // 30:30 "Enables debug mode"
  uint32_t reserved_3 : 1; // 31:31 "Reserved"
}odu_cfg_reg;

typedef struct {
  uint32_t be_size : 11; // 0:10 "Size of a single buffer element:\n 0: 16 Bytes\n 1: 32 Bytes\n …\n2047: 32768 Bytes"
}odu_be_size_reg;

typedef struct {
  uint32_t be_cnt : 11; // 0:10 "Number of buffer elements inside a buffer:\n 0: 1 BE\n 1: 2 BEs\n …\n2047: 2048 BEs"
}odu_be_cnt_reg;

typedef struct {
  uint32_t size : 9; // 0:8 "Storage element size in multiples of 16 tensor points:\n 0: 16 points\n 1: 32 points\n …\n 511: 8192 points"
}se_size_reg;

typedef struct {
  uint32_t y : 13; // 0:12 "Output tensor X dimension:\n 0: 1 point\n 1: 2 points\n ..\n 8191: 8192 points"
  uint32_t z : 13; // 13:25 "Output tensor Z dimension:\n 0: 1 point\n 1: 2 points\n ..\n 8191: 8192 points"
}te_dim0_reg;

typedef struct {
  uint32_t x : 13; // 0:12 "Output tensor Z dimension:\n 0: 1 point\n 1: 2 points\n ..\n 8191: 8192 points"
}te_dim1_reg;

typedef struct {
  uint32_t pt_base; // 0:31 "Output tensor storage element array base pointer"
}pt_base_reg;

typedef struct {
  uint32_t sp_base; // 0:31 "Output tensor sparsity base pointer"
}sp_base_reg;

typedef struct {
  uint32_t base_ptr_1 : 9; // 0:8 "ODU base pointer 1 (RESERVED)"
  uint32_t base_ptr_0 : 9; // 9:17 "ODU base pointer 0 (RESERVED)"
}base_ptr_a_reg;

typedef struct {
  uint32_t base_ptr_1 : 9; // 0:8 "ODU base pointer 3 (RESERVED)"
  uint32_t base_ptr_0 : 9; // 9:17 "ODU base pointer 2 (RESERVED)"
}base_ptr_b_reg;

typedef struct {
  uint32_t unused0 : 4;
  uint32_t base_adr : 28; // 4:31 "ODU base address 0 (RESERVED)"
}base_adr_0_reg;

typedef struct {
  uint32_t unused0 : 4;
  uint32_t base_adr : 28; // 4:31 "ODU base address 1 (RESERVED)"
}base_adr_1_reg;

typedef struct {
  uint32_t unused0 : 4;
  uint32_t base_adr : 28; // 4:31 "ODU base address 2 (RESERVED)"
}base_adr_2_reg;

typedef struct {
  uint32_t unused0 : 4;
  uint32_t base_adr : 28; // 4:31 "ODU base address 3 (RESERVED)"
}base_adr_3_reg;

typedef struct {
  uint32_t cast_enable : 1; // 0:0 "ODU cast enable 0"
  uint32_t unused0 : 3;
  uint32_t cast_offset : 28; // 4:31 "ODU cast offset 0"
}odu_cast_0_reg;

typedef struct {
  uint32_t cast_enable : 1; // 0:0 "ODU cast enable 1"
  uint32_t unused0 : 3;
  uint32_t cast_offset : 28; // 4:31 "ODU cast offset 1"
}odu_cast_1_reg;

typedef struct {
  uint32_t cast_enable : 1; // 0:0 "ODU cast enable 2"
  uint32_t unused0 : 3;
  uint32_t cast_offset : 28; // 4:31 "ODU cast offset 2"
}odu_cast_2_reg;

typedef struct {
  uint32_t mpe_wtbias : 8; // 0:7 "Weight bias - this is subtracted from all weights in the MPE when the data type is U8. It is unused for all other data types."
  uint32_t mpe_actbias : 8; // 8:15 "Activation bias - this is subtracted from all activations in the MPE when the data type is U8. It is unused for all other data types."
  uint32_t mpe_mode : 3; // 16:18 "UNUSED IN VPU2P7\n\nActivation/weight data type.\n 000 - FP16\n 001 - U8\n 010 - I8\n 011 - I4\n 100 - I2\n 111 - BIN\nWhen the activations and weights are of different types, MPE_MODE must be configured to the larger of the 2 data types."
  uint32_t mpe_dense : 1; // 19:19 "UNUSED IN VPU2P7\n\nWhen set enables a dense mode power optimisation in the MPEs - the power optimisation can only be used in ZM mode when padding is disabled."
  uint32_t mrm_weight_dense : 1; // 20:20 "UNUSED IN VPU2P7\n\nWhen set enables a dense mode power optimisation in the weight MRM - the power optimisation can only be used in ZM mode when padding is disabled."
  uint32_t mrm_act_dense : 1; // 21:21 "UNUSED IN VPU2P7\n\nWhen set enables a dense mode power optimisation in the activation MRM - the power optimisation can only be used in ZM mode when padding is disabled"
  uint32_t mpe_daz : 1; // 22:22 "1 - mpe will force denormal operands to zero\n0 - mpe will not force denormal operands to zero"
  uint32_t mpe_ftz : 1; // 23:23 "1 - mpe will force denormal results to zero\n0 - mpe will not force denormal results to zero"
}mpe_cfg_reg;

typedef struct {
  uint32_t mpe_bus_data_sel : 8; // 0:7 "Selects which of the 256 MPE's is selected for a bus data operation"
  uint32_t mpe_bus_data_en : 1; // 8:8 "Bus data enable"
}mpe_bus_data_sel_reg;

typedef struct {
  uint32_t elop_scale_b : 16; // 0:15 "Scale multiplier (u16) applied to tensor B. PPE shift is applied post-elop"
  uint32_t elop_scale_a : 16; // 16:31 "Scale multiplier (u16) applied to tensor A. PPE shift is applied post-elop"
}elop_scale_reg;

typedef struct {
  uint32_t ppe_g8_bias_c : 9; // 0:8 "i9 output zero-point offset (offset is expected to be within the range -128 to 256)"
  uint32_t ppe_g8_bias_b : 9; // 9:17 "i9 input B zero-point offset (offset is expected to be within the range -128 to 256)"
  uint32_t ppe_g8_bias_a : 9; // 18:26 "i9 input A zero-point offset (offset is expected to be within the range -128 to 256)"
}ppe_cfg_reg;

typedef struct {
  uint32_t ppe_bias; // 0:31 "scale bias (i32)"
}ppe_bias_reg;

typedef struct {
  uint32_t unused0 : 2;
  uint32_t ppe_scale_shift : 6; // 2:7 "scale shift (u6)"
  uint32_t unused1 : 2;
  uint32_t ppe_scale_round : 2; // 10:11 "scale round (00 - Round to nearest ties to even, 01 - Unused, 10 - Round to nearest ties away from zero, 11 - no round)"
  uint32_t unused2 : 4;
  uint32_t ppe_scale_mult : 16; // 16:31 "scale multiplier[15:0] (i16)"
}ppe_scale_reg;

typedef struct {
  uint32_t ppe_scale_override : 1; // 0:0 "defines if the scale is taken from the registers or from the scale table"
  uint32_t ppe_fp_scale_override : 1; // 1:1 "defines if the floating-point scale is taken from the registers or from the scale table"
}ppe_scale_ctrl_reg;

typedef struct {
  uint32_t unused0 : 8;
  uint32_t ppe_prelu_shift : 5; // 8:12 "prelu shift (u5)"
  uint32_t unused1 : 3;
  uint32_t ppe_prelu_mult : 11; // 16:26 "prelu multiplier (u11)"
}ppe_prelu_reg;

typedef struct {
  uint32_t vpu2p0_rsvd_1 : 1; // 0:0 "Not used"
}vpu2p0_rsvd_1_reg;

typedef struct {
  uint32_t vpu2p0_rsvd_2 : 1; // 0:0 "Not used"
}vpu2p0_rsvd_2_reg;

typedef struct {
  uint32_t ppe_scale_hclamp; // 0:31 "PPE scale high clamp is used to clamp the scale pipe"
}ppe_scale_hclamp_reg;

typedef struct {
  uint32_t ppe_scale_lclamp; // 0:31 "PPE scale low clamp is used to clamp the scale pipe"
}ppe_scale_lclamp_reg;

typedef struct {
  uint32_t vpu2p0_rsvd_3 : 1; // 0:0 "Not used"
}vpu2p0_rsvd_3_reg;

typedef struct {
  uint32_t unused0 : 6;
  uint32_t ppe_fp16_ftz : 1; // 6:6 "fp32 to fp16 conversion Denormal force-to-zero\n1 - Force denorms to zero\n0 - Support denorms (ie Don't force to zero)"
  uint32_t ppe_fp16_clamp : 1; // 7:7 "fp32 to fp16 conversion clamping\n1 - Clamp to max/min fp16 on overflow\n0 - No clamping (normal overflow behaviour)"
  uint32_t ppe_i32_convert : 2; // 8:9 "Enables conversion of s17.15 to fp16 or fp8\n00 - No conversion\n01 - convert s17.15 to fp16\n10 - convert s17.15 to fp8 (conversion to fp16, followed by truncation)\n11 - No conversion"
}ppe_misc_reg;

typedef struct {
  uint32_t ppe_fp_bias; // 0:31 "scale bias (fp32)"
}ppe_fp_bias_reg;

typedef struct {
  uint32_t ppe_fp_scale; // 0:31 "scale multiplier (fp32)"
}ppe_fp_scale_reg;

typedef struct {
  uint32_t ppe_fp_prelu; // 0:31 "prelu multiplier (fp32)"
}ppe_fp_prelu_reg;

typedef struct {
  uint32_t ppe_fp_convert : 3; // 0:2 "FP conversion (000 - none, 001 - FP16, 010 - BF16, 011 - FP8, 100 - i32)"
  uint32_t ppe_fp_bypass : 1; // 3:3 "bypass FP32 adder (bias) and FP32 multiplier (scale)"
  uint32_t ppe_bf16_round : 1; // 4:4 "Rounding mode used in FP32 to BF16 conversion. 0 - No rounding (truncation), 1 - RNE"
  uint32_t ppe_fp_prelu_en : 1; // 5:5 "0 - Disable FP prelu, 1 - Enable FP prelu"
}ppe_fp_cfg_reg;

typedef struct {
  uint32_t odu2_int_sts; // 0:31 "odu interrupt status"
}odu2_int_sts_reg;

typedef struct {
  uint32_t odu2_int_raw_sts; // 0:31 "odu raw interrupt status"
}odu2_int_raw_sts_reg;

typedef struct {
  uint32_t odu2_int_en; // 0:31 "odu interupt enable"
}odu2_int_en_sts_reg;

typedef struct {
  uint32_t odu2_int_clr_0 : 1; // 0:0 "odu interupt clear"
  uint32_t odu2_int_clr_1 : 1; // 1:1 "odu interupt clear"
  uint32_t odu2_int_clr_2 : 1; // 2:2 "odu interupt clear"
  uint32_t odu2_int_clr_3 : 1; // 3:3 "odu interupt clear"
  uint32_t odu2_int_clr_4 : 1; // 4:4 "odu interupt clear"
  uint32_t odu2_int_clr_5 : 1; // 5:5 "odu interupt clear"
  uint32_t odu2_int_clr_6 : 1; // 6:6 "odu interupt clear"
  uint32_t odu2_int_clr_7 : 1; // 7:7 "odu interupt clear"
  uint32_t odu2_int_clr_8 : 1; // 8:8 "odu interupt clear"
  uint32_t odu2_int_clr_9 : 1; // 9:9 "odu interupt clear"
  uint32_t odu2_int_clr_10 : 1; // 10:10 "odu interupt clear"
  uint32_t odu2_int_clr_11 : 1; // 11:11 "odu interupt clear"
  uint32_t odu2_int_clr_12 : 1; // 12:12 "odu interupt clear"
  uint32_t odu2_int_clr_13 : 1; // 13:13 "odu interupt clear"
  uint32_t odu2_int_clr_14 : 1; // 14:14 "odu interupt clear"
  uint32_t odu2_int_clr_15 : 1; // 15:15 "odu interupt clear"
  uint32_t odu2_int_clr_16 : 1; // 16:16 "odu interupt clear"
  uint32_t odu2_int_clr_17 : 1; // 17:17 "odu interupt clear"
  uint32_t odu2_int_clr_18 : 1; // 18:18 "odu interupt clear"
  uint32_t odu2_int_clr_19 : 1; // 19:19 "odu interupt clear"
  uint32_t odu2_int_clr_20 : 1; // 20:20 "odu interupt clear"
  uint32_t odu2_int_clr_21 : 1; // 21:21 "odu interupt clear"
  uint32_t odu2_int_clr_22 : 1; // 22:22 "odu interupt clear"
  uint32_t odu2_int_clr_23 : 1; // 23:23 "odu interupt clear"
  uint32_t odu2_int_clr_24 : 1; // 24:24 "odu interupt clear"
  uint32_t odu2_int_clr_25 : 1; // 25:25 "odu interupt clear"
  uint32_t odu2_int_clr_26 : 1; // 26:26 "odu interupt clear"
  uint32_t odu2_int_clr_27 : 1; // 27:27 "odu interupt clear"
  uint32_t odu2_int_clr_28 : 1; // 28:28 "odu interupt clear"
  uint32_t odu2_int_clr_29 : 1; // 29:29 "odu interupt clear"
  uint32_t odu2_int_clr_30 : 1; // 30:30 "odu interupt clear"
  uint32_t odu2_int_clr_31 : 1; // 31:31 "odu interupt clear"
}odu2_int_clr_reg;

typedef struct {
  uint32_t mpe_int_sts; // 0:31 "MPE interrupt status"
}mpe_int_sts_reg;

typedef struct {
  uint32_t mpe_int_raw_sts; // 0:31 "MPE raw interrupt status"
}mpe_int_raw_sts_reg;

typedef struct {
  uint32_t mpe_int_en; // 0:31 "MPE interupt enable"
}mpe_int_en_sts_reg;

typedef struct {
  uint32_t mpe_int_clr_0 : 1; // 0:0 "MPE interupt clear"
  uint32_t mpe_int_clr_1 : 1; // 1:1 "MPE interupt clear"
  uint32_t mpe_int_clr_2 : 1; // 2:2 "MPE interupt clear"
  uint32_t mpe_int_clr_3 : 1; // 3:3 "MPE interupt clear"
  uint32_t mpe_int_clr_4 : 1; // 4:4 "MPE interupt clear"
  uint32_t mpe_int_clr_5 : 1; // 5:5 "MPE interupt clear"
  uint32_t mpe_int_clr_6 : 1; // 6:6 "MPE interupt clear"
  uint32_t mpe_int_clr_7 : 1; // 7:7 "MPE interupt clear"
  uint32_t mpe_int_clr_8 : 1; // 8:8 "MPE interupt clear"
  uint32_t mpe_int_clr_9 : 1; // 9:9 "MPE interupt clear"
  uint32_t mpe_int_clr_10 : 1; // 10:10 "MPE interupt clear"
  uint32_t mpe_int_clr_11 : 1; // 11:11 "MPE interupt clear"
  uint32_t mpe_int_clr_12 : 1; // 12:12 "MPE interupt clear"
  uint32_t mpe_int_clr_13 : 1; // 13:13 "MPE interupt clear"
  uint32_t mpe_int_clr_14 : 1; // 14:14 "MPE interupt clear"
  uint32_t mpe_int_clr_15 : 1; // 15:15 "MPE interupt clear"
  uint32_t mpe_int_clr_16 : 1; // 16:16 "MPE interupt clear"
  uint32_t mpe_int_clr_17 : 1; // 17:17 "MPE interupt clear"
  uint32_t mpe_int_clr_18 : 1; // 18:18 "MPE interupt clear"
  uint32_t mpe_int_clr_19 : 1; // 19:19 "MPE interupt clear"
  uint32_t mpe_int_clr_20 : 1; // 20:20 "MPE interupt clear"
  uint32_t mpe_int_clr_21 : 1; // 21:21 "MPE interupt clear"
  uint32_t mpe_int_clr_22 : 1; // 22:22 "MPE interupt clear"
  uint32_t mpe_int_clr_23 : 1; // 23:23 "MPE interupt clear"
  uint32_t mpe_int_clr_24 : 1; // 24:24 "MPE interupt clear"
  uint32_t mpe_int_clr_25 : 1; // 25:25 "MPE interupt clear"
  uint32_t mpe_int_clr_26 : 1; // 26:26 "MPE interupt clear"
  uint32_t mpe_int_clr_27 : 1; // 27:27 "MPE interupt clear"
  uint32_t mpe_int_clr_28 : 1; // 28:28 "MPE interupt clear"
  uint32_t mpe_int_clr_29 : 1; // 29:29 "MPE interupt clear"
  uint32_t mpe_int_clr_30 : 1; // 30:30 "MPE interupt clear"
  uint32_t mpe_int_clr_31 : 1; // 31:31 "MPE interupt clear"
}mpe_int_clr_reg;

typedef struct {
  uint32_t ppe_int_sts; // 0:31 "ppe interrupt status"
}ppe_int_sts_reg;

typedef struct {
  uint32_t ppe_int_raw_sts; // 0:31 "ppe raw interrupt status"
}ppe_int_raw_sts_reg;

typedef struct {
  uint32_t ppe_int_en; // 0:31 "ppe interupt enable"
}ppe_int_en_sts_reg;

typedef struct {
  uint32_t ppe_int_clr_0 : 1; // 0:0 "ppe interupt clear"
  uint32_t ppe_int_clr_1 : 1; // 1:1 "ppe interupt clear"
  uint32_t ppe_int_clr_2 : 1; // 2:2 "ppe interupt clear"
  uint32_t ppe_int_clr_3 : 1; // 3:3 "ppe interupt clear"
  uint32_t ppe_int_clr_4 : 1; // 4:4 "ppe interupt clear"
  uint32_t ppe_int_clr_5 : 1; // 5:5 "ppe interupt clear"
  uint32_t ppe_int_clr_6 : 1; // 6:6 "ppe interupt clear"
  uint32_t ppe_int_clr_7 : 1; // 7:7 "ppe interupt clear"
  uint32_t ppe_int_clr_8 : 1; // 8:8 "ppe interupt clear"
  uint32_t ppe_int_clr_9 : 1; // 9:9 "ppe interupt clear"
  uint32_t ppe_int_clr_10 : 1; // 10:10 "ppe interupt clear"
  uint32_t ppe_int_clr_11 : 1; // 11:11 "ppe interupt clear"
  uint32_t ppe_int_clr_12 : 1; // 12:12 "ppe interupt clear"
  uint32_t ppe_int_clr_13 : 1; // 13:13 "ppe interupt clear"
  uint32_t ppe_int_clr_14 : 1; // 14:14 "ppe interupt clear"
  uint32_t ppe_int_clr_15 : 1; // 15:15 "ppe interupt clear"
  uint32_t ppe_int_clr_16 : 1; // 16:16 "ppe interupt clear"
  uint32_t ppe_int_clr_17 : 1; // 17:17 "ppe interupt clear"
  uint32_t ppe_int_clr_18 : 1; // 18:18 "ppe interupt clear"
  uint32_t ppe_int_clr_19 : 1; // 19:19 "ppe interupt clear"
  uint32_t ppe_int_clr_20 : 1; // 20:20 "ppe interupt clear"
  uint32_t ppe_int_clr_21 : 1; // 21:21 "ppe interupt clear"
  uint32_t ppe_int_clr_22 : 1; // 22:22 "ppe interupt clear"
  uint32_t ppe_int_clr_23 : 1; // 23:23 "ppe interupt clear"
  uint32_t ppe_int_clr_24 : 1; // 24:24 "ppe interupt clear"
  uint32_t ppe_int_clr_25 : 1; // 25:25 "ppe interupt clear"
  uint32_t ppe_int_clr_26 : 1; // 26:26 "ppe interupt clear"
  uint32_t ppe_int_clr_27 : 1; // 27:27 "ppe interupt clear"
  uint32_t ppe_int_clr_28 : 1; // 28:28 "ppe interupt clear"
  uint32_t ppe_int_clr_29 : 1; // 29:29 "ppe interupt clear"
  uint32_t ppe_int_clr_30 : 1; // 30:30 "ppe interupt clear"
  uint32_t ppe_int_clr_31 : 1; // 31:31 "ppe interupt clear"
}ppe_int_clr_reg;

typedef struct {
  uint32_t idu_int_sts; // 0:31 "idu interrupt status"
}idu_int_sts_reg;

typedef struct {
  uint32_t idu_int_raw_sts; // 0:31 "idu raw interrupt status"
}idu_int_raw_sts_reg;

typedef struct {
  uint32_t idu_int_en; // 0:31 "idu interupt enable"
}idu_int_en_sts_reg;

typedef struct {
  uint32_t idu_int_clr_0 : 1; // 0:0 "idu interupt clear"
  uint32_t idu_int_clr_1 : 1; // 1:1 "idu interupt clear"
  uint32_t idu_int_clr_2 : 1; // 2:2 "idu interupt clear"
  uint32_t idu_int_clr_3 : 1; // 3:3 "idu interupt clear"
  uint32_t idu_int_clr_4 : 1; // 4:4 "idu interupt clear"
  uint32_t idu_int_clr_5 : 1; // 5:5 "idu interupt clear"
  uint32_t idu_int_clr_6 : 1; // 6:6 "idu interupt clear"
  uint32_t idu_int_clr_7 : 1; // 7:7 "idu interupt clear"
  uint32_t idu_int_clr_8 : 1; // 8:8 "idu interupt clear"
  uint32_t idu_int_clr_9 : 1; // 9:9 "idu interupt clear"
  uint32_t idu_int_clr_10 : 1; // 10:10 "idu interupt clear"
  uint32_t idu_int_clr_11 : 1; // 11:11 "idu interupt clear"
  uint32_t idu_int_clr_12 : 1; // 12:12 "idu interupt clear"
  uint32_t idu_int_clr_13 : 1; // 13:13 "idu interupt clear"
  uint32_t idu_int_clr_14 : 1; // 14:14 "idu interupt clear"
  uint32_t idu_int_clr_15 : 1; // 15:15 "idu interupt clear"
  uint32_t idu_int_clr_16 : 1; // 16:16 "idu interupt clear"
  uint32_t idu_int_clr_17 : 1; // 17:17 "idu interupt clear"
  uint32_t idu_int_clr_18 : 1; // 18:18 "idu interupt clear"
  uint32_t idu_int_clr_19 : 1; // 19:19 "idu interupt clear"
  uint32_t idu_int_clr_20 : 1; // 20:20 "idu interupt clear"
  uint32_t idu_int_clr_21 : 1; // 21:21 "idu interupt clear"
  uint32_t idu_int_clr_22 : 1; // 22:22 "idu interupt clear"
  uint32_t idu_int_clr_23 : 1; // 23:23 "idu interupt clear"
  uint32_t idu_int_clr_24 : 1; // 24:24 "idu interupt clear"
  uint32_t idu_int_clr_25 : 1; // 25:25 "idu interupt clear"
  uint32_t idu_int_clr_26 : 1; // 26:26 "idu interupt clear"
  uint32_t idu_int_clr_27 : 1; // 27:27 "idu interupt clear"
  uint32_t idu_int_clr_28 : 1; // 28:28 "idu interupt clear"
  uint32_t idu_int_clr_29 : 1; // 29:29 "idu interupt clear"
  uint32_t idu_int_clr_30 : 1; // 30:30 "idu interupt clear"
  uint32_t idu_int_clr_31 : 1; // 31:31 "idu interupt clear"
}idu_int_clr_reg;

typedef struct {
  uint32_t odu_int_sts; // 0:31 "odu interrupt status"
}odu_int_sts_reg;

typedef struct {
  uint32_t odu_int_raw_sts; // 0:31 "odu raw interrupt status"
}odu_int_raw_sts_reg;

typedef struct {
  uint32_t odu_int_en; // 0:31 "odu interupt enable"
}odu_int_en_sts_reg;

typedef struct {
  uint32_t odu_int_clr_0 : 1; // 0:0 "odu interupt clear"
  uint32_t odu_int_clr_1 : 1; // 1:1 "odu interupt clear"
  uint32_t odu_int_clr_2 : 1; // 2:2 "odu interupt clear"
  uint32_t odu_int_clr_3 : 1; // 3:3 "odu interupt clear"
  uint32_t odu_int_clr_4 : 1; // 4:4 "odu interupt clear"
  uint32_t odu_int_clr_5 : 1; // 5:5 "odu interupt clear"
  uint32_t odu_int_clr_6 : 1; // 6:6 "odu interupt clear"
  uint32_t odu_int_clr_7 : 1; // 7:7 "odu interupt clear"
  uint32_t odu_int_clr_8 : 1; // 8:8 "odu interupt clear"
  uint32_t odu_int_clr_9 : 1; // 9:9 "odu interupt clear"
  uint32_t odu_int_clr_10 : 1; // 10:10 "odu interupt clear"
  uint32_t odu_int_clr_11 : 1; // 11:11 "odu interupt clear"
  uint32_t odu_int_clr_12 : 1; // 12:12 "odu interupt clear"
  uint32_t odu_int_clr_13 : 1; // 13:13 "odu interupt clear"
  uint32_t odu_int_clr_14 : 1; // 14:14 "odu interupt clear"
  uint32_t odu_int_clr_15 : 1; // 15:15 "odu interupt clear"
  uint32_t odu_int_clr_16 : 1; // 16:16 "odu interupt clear"
  uint32_t odu_int_clr_17 : 1; // 17:17 "odu interupt clear"
  uint32_t odu_int_clr_18 : 1; // 18:18 "odu interupt clear"
  uint32_t odu_int_clr_19 : 1; // 19:19 "odu interupt clear"
  uint32_t odu_int_clr_20 : 1; // 20:20 "odu interupt clear"
  uint32_t odu_int_clr_21 : 1; // 21:21 "odu interupt clear"
  uint32_t odu_int_clr_22 : 1; // 22:22 "odu interupt clear"
  uint32_t odu_int_clr_23 : 1; // 23:23 "odu interupt clear"
  uint32_t odu_int_clr_24 : 1; // 24:24 "odu interupt clear"
  uint32_t odu_int_clr_25 : 1; // 25:25 "odu interupt clear"
  uint32_t odu_int_clr_26 : 1; // 26:26 "odu interupt clear"
  uint32_t odu_int_clr_27 : 1; // 27:27 "odu interupt clear"
  uint32_t odu_int_clr_28 : 1; // 28:28 "odu interupt clear"
  uint32_t odu_int_clr_29 : 1; // 29:29 "odu interupt clear"
  uint32_t odu_int_clr_30 : 1; // 30:30 "odu interupt clear"
  uint32_t odu_int_clr_31 : 1; // 31:31 "odu interupt clear"
}odu_int_clr_reg;

typedef struct {
  uint32_t fifo_write : 1; // 0:0 "Writing 1 to this field, pushes ac_base into the FIFO\nReading 1 from this field, indicates that the FIFO is full"
  uint32_t fifo_clear : 1; // 1:1 "Writing 1 to fhis field, initiates clearing of the FIFO\nReading 1 from this field, indicates clearing of the FIFO in progress"
  uint32_t buffer_reset : 1; // 2:2 "Writing 1 to this field, clears all buffer assigments\nReading 1 from this field, indicates buffer clearing in progress"
  uint32_t unused0 : 1;
  uint32_t ac_base : 28; // 4:31 "Activation base address"
}odu_ac_base_reg;

typedef struct {
  uint32_t hwp_en : 1; // 0:0 "Enable HW profiler and It works in conjunction with idu and odu stat enable"
}hwp_ctrl_reg;

typedef struct {
  uint32_t cfg_wload_id : 16; // 0:15 "Setting up workload ID"
}hwp_wload_id_reg;

typedef struct {
  uint32_t hwp_cmx_mem_addr : 28; // 0:27 "hwp cmx pointer. Only addr[27:0] is being used"
}hwp_cmx_mem_adr_reg;

typedef struct {
  uint32_t workload_duration; // 0:31 "Statistic for the number of clock cycles required to process the workload"
}idu_wl_stat_reg;

typedef struct {
  uint32_t sparse_act; // 0:31 "Statistic for the number of sparse activations in the current workload"
}idu_sa_stat_reg;

typedef struct {
  uint32_t sparse_wt; // 0:31 "Statistic for the number of sparse weights in the current workload"
}idu_sw_stat_reg;

typedef struct {
  uint32_t dense_act; // 0:31 "Statistic for the number of dense activations in the current workload"
}idu_da_stat_reg;

typedef struct {
  uint32_t dense_wt; // 0:31 "Statistic for the number of dense weights in the current workload"
}idu_dw_stat_reg;

typedef struct {
  uint32_t cmx0_stall; // 0:31 "Statistic for the number of stalls on CMX I/F0"
}idu_cmx0_st_stat_reg;

typedef struct {
  uint32_t cmx1_stall; // 0:31 "Statistic for the number of stalls on CMX I/F1"
}idu_cmx1_st_stat_reg;

typedef struct {
  uint32_t cmx2_stall; // 0:31 "Statistic for the number of stalls on CMX I/F2"
}idu_cmx2_st_stat_reg;

typedef struct {
  uint32_t cmx3_stall; // 0:31 "Statistic for the number of stalls on CMX I/F3"
}idu_cmx3_st_stat_reg;

typedef struct {
  uint32_t cmx0_read; // 0:31 "Statistic for the number of reads on CMX I/F0"
}idu_cmx0_rd_stat_reg;

typedef struct {
  uint32_t cmx1_read; // 0:31 "Statistic for the number of reads on CMX I/F1"
}idu_cmx1_rd_stat_reg;

typedef struct {
  uint32_t cmx2_read; // 0:31 "Statistic for the number of reads on CMX I/F2"
}idu_cmx2_rd_stat_reg;

typedef struct {
  uint32_t cmx3_read; // 0:31 "Statistic for the number of reads on CMX I/F3"
}idu_cmx3_rd_stat_reg;

typedef struct {
  uint32_t cmx4_stall; // 0:31 "Statistic for the number of stalls on CMX I/F4"
}idu_cmx4_st_stat_reg;

typedef struct {
  uint32_t cmx5_stall; // 0:31 "Statistic for the number of stalls on CMX I/F5"
}idu_cmx5_st_stat_reg;

typedef struct {
  uint32_t cmx6_stall; // 0:31 "Statistic for the number of stalls on CMX I/F6"
}idu_cmx6_st_stat_reg;

typedef struct {
  uint32_t cmx7_stall; // 0:31 "Statistic for the number of stalls on CMX I/F7"
}idu_cmx7_st_stat_reg;

typedef struct {
  uint32_t cmx4_read; // 0:31 "Statistic for the number of reads on CMX I/F4"
}idu_cmx4_rd_stat_reg;

typedef struct {
  uint32_t cmx5_read; // 0:31 "Statistic for the number of reads on CMX I/F5"
}idu_cmx5_rd_stat_reg;

typedef struct {
  uint32_t cmx6_read; // 0:31 "Statistic for the number of reads on CMX I/F6"
}idu_cmx6_rd_stat_reg;

typedef struct {
  uint32_t cmx7_read; // 0:31 "Statistic for the number of reads on CMX I/F7"
}idu_cmx7_rd_stat_reg;

typedef struct {
  uint32_t stat_valid : 1; // 0:0 "ODU statistics valid (RESERVED)"
}odu_stat_reg;

typedef struct {
  uint32_t stat_sp_1_cnt; // 0:31 "Non-sparse point count"
}odu_stat_one_cnt_reg;

typedef struct {
  uint32_t stat_sp_0_cnt; // 0:31 "Sparse point count"
}odu_stat_zero_cnt_reg;

typedef struct {
  uint32_t stat_stall_cnt_0; // 0:31 "Number of stall events on CMX interface 0. Saturates at 0xFFFF."
}odu_stat_stall_0_reg;

typedef struct {
  uint32_t stat_stall_cnt_1; // 0:31 "Number of stall events on CMX interface 1. Saturates at 0xFFFF."
}odu_stat_stall_1_reg;

typedef struct {
  uint32_t stat_stall_cnt_2; // 0:31 "Number of stall events on CMX interface 2. Saturates at 0xFFFF."
}odu_stat_stall_2_reg;

typedef struct {
  uint32_t stat_stall_cnt_3; // 0:31 "Number of stall events on CMX interface 3. Saturates at 0xFFFF."
}odu_stat_stall_3_reg;

typedef struct {
  uint32_t stat_clk_cycle_cnt; // 0:31 "Number of clock cycles spent processing workload. Saturates at 0xFFFF_FFFF."
}odu_clk_cycle_cnt_reg;

typedef struct {
  uint32_t idu_debug; // 0:31 "Debug register shared between the ZM, CM and DW blocks."
}idu_debug_reg;

typedef struct {
  uint32_t hsspsram_rm : 4; // 0:3 "RM"
  uint32_t hsspsram_ra : 2; // 4:5 "RA (unused)"
  uint32_t hsspsram_wa : 3; // 6:8 "WA (unused)"
  uint32_t hsspsram_wpulse : 3; // 9:11 "WPULSE (unused)"
  uint32_t hsspsram_rme : 1; // 12:12 "RME"
  uint32_t hsspsram_ls : 1; // 13:13 "LS"
  uint32_t hsspsram_ds : 1; // 14:14 "DS"
  uint32_t hsspsram_sd : 1; // 15:15 "SD (unused)"
  uint32_t hsspsram_reserved : 16; // 16:31 "Reserved"
}hsspsram_reg;

typedef struct {
  uint32_t hdspsram_rm : 4; // 0:3 "RM"
  uint32_t hdspsram_ra : 2; // 4:5 "RA (unused)"
  uint32_t hdspsram_wa : 3; // 6:8 "WA (unused)"
  uint32_t hdspsram_wpulse : 3; // 9:11 "WPULSE (unused)"
  uint32_t hdspsram_rme : 1; // 12:12 "RME"
  uint32_t hdspsram_ls : 1; // 13:13 "LS"
  uint32_t hdspsram_ds : 1; // 14:14 "DS"
  uint32_t hdspsram_sd : 1; // 15:15 "SD (unused)"
  uint32_t hdspsram_reserved : 16; // 16:31 "Reserved"
}hdspsram_reg;

typedef struct {
  uint32_t uhdspsram_rm : 4; // 0:3 "RM"
  uint32_t uhdspsram_ra : 2; // 4:5 "RA (unused)"
  uint32_t uhdspsram_wa : 3; // 6:8 "WA (unused)"
  uint32_t uhdspsram_wpulse : 3; // 9:11 "WPULSE (unused)"
  uint32_t uhdspsram_rme : 1; // 12:12 "RME"
  uint32_t uhdspsram_ls : 1; // 13:13 "LS"
  uint32_t uhdspsram_ds : 1; // 14:14 "DS"
  uint32_t uhdspsram_sd : 1; // 15:15 "SD (unused)"
  uint32_t uhdspsram_reserved : 16; // 16:31 "Reserved"
}uhdspsram_reg;

typedef struct {
  uint32_t hd1prf_rm : 4; // 0:3 "RM"
  uint32_t hd1prf_ra : 2; // 4:5 "RA (unused)"
  uint32_t hd1prf_wa : 3; // 6:8 "WA (unused)"
  uint32_t hd1prf_wpulse : 3; // 9:11 "WPULSE (unused)"
  uint32_t hd1prf_rme : 1; // 12:12 "RME"
  uint32_t hd1prf_ls : 1; // 13:13 "LS"
  uint32_t hd1prf_ds : 1; // 14:14 "DS"
  uint32_t hd1prf_sd : 1; // 15:15 "SD (unused)"
  uint32_t hd1prf_reserved : 16; // 16:31 "Reserved"
}hd1prf_reg;

typedef struct {
  uint32_t hd2prf_rm : 4; // 0:3 "RM"
  uint32_t hd2prf_ra : 2; // 4:5 "RA (unused)"
  uint32_t hd2prf_wa : 3; // 6:8 "WA (unused)"
  uint32_t hd2prf_wpulse : 3; // 9:11 "WPULSE (unused)"
  uint32_t hd2prf_rme : 1; // 12:12 "RME"
  uint32_t hd2prf_ls : 1; // 13:13 "LS"
  uint32_t hd2prf_ds : 1; // 14:14 "DS"
  uint32_t hd2prf_sd : 1; // 15:15 "SD (unused)"
  uint32_t hd2prf_reserved : 16; // 16:31 "Reserved"
}hd2prf_reg;

typedef struct {
  uint32_t uhd2prf_rm : 4; // 0:3 "RM"
  uint32_t uhd2prf_ra : 2; // 4:5 "RA (unused)"
  uint32_t uhd2prf_wa : 3; // 6:8 "WA (unused)"
  uint32_t uhd2prf_wpulse : 3; // 9:11 "WPULSE (unused)"
  uint32_t uhd2prf_rme : 1; // 12:12 "RME"
  uint32_t uhd2prf_ls : 1; // 13:13 "LS"
  uint32_t uhd2prf_ds : 1; // 14:14 "DS"
  uint32_t uhd2prf_sd : 1; // 15:15 "SD (unused)"
  uint32_t uhd2prf_reserved : 16; // 16:31 "Reserved"
}uhd2prf_reg;


typedef struct {
  dpu_cfg_reg dpu_cfg;
  dpu_sts_reg dpu_sts;
  vpu2p0_rsvd_0_reg vpu2p0_rsvd_0;
  uint32_t unused0[5];
  workload_size0_reg workload_size0;
  workload_size1_reg workload_size1;
  workload_start0_reg workload_start0;
  workload_start1_reg workload_start1;
  compute_stencil_reg compute_stencil;
  te_end0_reg te_end0;
  te_end1_reg te_end1;
  te_beg0_reg te_beg0;
  te_beg1_reg te_beg1;
} cfg_top_description;

typedef struct {
  uint32_t se_addr;
  uint32_t sparsity_addr;
  uint32_t se_addr1;
  uint32_t sparsity_addr1;
  uint32_t se_addr2;
  uint32_t sparsity_addr2;
  uint32_t se_addr3;
  uint32_t sparsity_addr3;
  se_sp_size_reg se_sp_size;
  se_sp_size1_reg se_sp_size1;
  se_sp_size2_reg se_sp_size2;
  z_config_reg z_config;
  kernel_pad_cfg_reg kernel_pad_cfg;
  weight_size_reg weight_size;
  weight_num_reg weight_num;
  uint32_t weight_start;
  tensor_size0_reg tensor_size0;
  tensor_size1_reg tensor_size1;
  uint32_t tensor_start;
  tensor_mode_reg tensor_mode;
  uint32_t elops_sparsity_addr;
  uint32_t elops_se_addr;
  elops_wload_reg elops_wload;
  uint32_t act0_offset;
  uint32_t act1_offset;
  uint32_t act2_offset;
  uint32_t act3_offset;
  base_offseta_reg base_offseta;
  base_offsetb_reg base_offsetb;
  uint32_t wt_offset;
} cfg_idu_description;

typedef struct {
  odu_cfg_reg odu_cfg;
  odu_be_size_reg odu_be_size;
  odu_be_cnt_reg odu_be_cnt;
  se_size_reg se_size;
  te_dim0_reg te_dim0;
  te_dim1_reg te_dim1;
  uint32_t pt_base;
  uint32_t sp_base;
  base_ptr_a_reg base_ptr_a;
  base_ptr_b_reg base_ptr_b;
  base_adr_0_reg base_adr_0;
  base_adr_1_reg base_adr_1;
  base_adr_2_reg base_adr_2;
  base_adr_3_reg base_adr_3;
  odu_cast_0_reg odu_cast_0;
  odu_cast_1_reg odu_cast_1;
  odu_cast_2_reg odu_cast_2;
} cfg_odu_description;

typedef struct {
  cfg_top_description top;
  uint32_t unused0[3];
  cmx_slice0_low_addr_reg cmx_slice0_low_addr;
  cmx_slice1_low_addr_reg cmx_slice1_low_addr;
  cmx_slice2_low_addr_reg cmx_slice2_low_addr;
  cmx_slice3_low_addr_reg cmx_slice3_low_addr;
  cmx_slice_size_reg cmx_slice_size;
  cmx_slice0_high_addr_reg cmx_slice0_high_addr;
  cmx_slice1_high_addr_reg cmx_slice1_high_addr;
  cmx_slice2_high_addr_reg cmx_slice2_high_addr;
  cmx_slice3_high_addr_reg cmx_slice3_high_addr;
  cmx_slice_enable_status_reg cmx_slice_enable_status;
  uint32_t unused1[2];
  cfg_idu_description idu;
  cfg_odu_description odu;
  mpe_cfg_reg mpe_cfg;
  mpe_bus_data_sel_reg mpe_bus_data_sel;
  elop_scale_reg elop_scale;
  ppe_cfg_reg ppe_cfg;
  uint32_t ppe_bias;
  ppe_scale_reg ppe_scale;
  ppe_scale_ctrl_reg ppe_scale_ctrl;
  ppe_prelu_reg ppe_prelu;
  vpu2p0_rsvd_1_reg vpu2p0_rsvd_1;
  vpu2p0_rsvd_2_reg vpu2p0_rsvd_2;
  uint32_t ppe_scale_hclamp;
  uint32_t ppe_scale_lclamp;
  vpu2p0_rsvd_3_reg vpu2p0_rsvd_3;
  ppe_misc_reg ppe_misc;
  uint32_t ppe_fp_bias;
  uint32_t ppe_fp_scale;
  uint32_t ppe_fp_prelu;
  ppe_fp_cfg_reg ppe_fp_cfg;
  uint32_t unused2[11];
  uint32_t odu2_int_sts;
  uint32_t odu2_int_raw_sts;
  uint32_t odu2_int_en_sts;
  odu2_int_clr_reg odu2_int_clr;
  uint32_t mpe_int_sts;
  uint32_t mpe_int_raw_sts;
  uint32_t mpe_int_en_sts;
  mpe_int_clr_reg mpe_int_clr;
  uint32_t ppe_int_sts;
  uint32_t ppe_int_raw_sts;
  uint32_t ppe_int_en_sts;
  ppe_int_clr_reg ppe_int_clr;
  uint32_t idu_int_sts;
  uint32_t idu_int_raw_sts;
  uint32_t idu_int_en_sts;
  idu_int_clr_reg idu_int_clr;
  uint32_t odu_int_sts;
  uint32_t odu_int_raw_sts;
  uint32_t odu_int_en_sts;
  odu_int_clr_reg odu_int_clr;
  odu_ac_base_reg odu_ac_base;
  uint32_t unused3[21];
  hwp_ctrl_reg hwp_ctrl;
  hwp_wload_id_reg hwp_wload_id;
  hwp_cmx_mem_adr_reg hwp_cmx_mem_adr;
  uint32_t unused4[7];
  uint32_t idu_wl_stat;
  uint32_t idu_sa_stat;
  uint32_t idu_sw_stat;
  uint32_t idu_da_stat;
  uint32_t idu_dw_stat;
  uint32_t idu_cmx0_st_stat;
  uint32_t idu_cmx1_st_stat;
  uint32_t idu_cmx2_st_stat;
  uint32_t idu_cmx3_st_stat;
  uint32_t idu_cmx0_rd_stat;
  uint32_t idu_cmx1_rd_stat;
  uint32_t idu_cmx2_rd_stat;
  uint32_t idu_cmx3_rd_stat;
  uint32_t idu_cmx4_st_stat;
  uint32_t idu_cmx5_st_stat;
  uint32_t idu_cmx6_st_stat;
  uint32_t idu_cmx7_st_stat;
  uint32_t idu_cmx4_rd_stat;
  uint32_t idu_cmx5_rd_stat;
  uint32_t idu_cmx6_rd_stat;
  uint32_t idu_cmx7_rd_stat;
  odu_stat_reg odu_stat;
  uint32_t odu_stat_one_cnt;
  uint32_t odu_stat_zero_cnt;
  uint32_t odu_stat_stall_0;
  uint32_t odu_stat_stall_1;
  uint32_t odu_stat_stall_2;
  uint32_t odu_stat_stall_3;
  uint32_t odu_clk_cycle_cnt;
  uint32_t idu_debug;
  uint32_t unused5[2];
  hsspsram_reg hsspsram;
  hdspsram_reg hdspsram;
  uhdspsram_reg uhdspsram;
  uint32_t unused6[1];
  hd1prf_reg hd1prf;
  hd2prf_reg hd2prf;
  uint32_t unused7[1];
  uhd2prf_reg uhd2prf;
} cfg_dpu_description;
#endif
