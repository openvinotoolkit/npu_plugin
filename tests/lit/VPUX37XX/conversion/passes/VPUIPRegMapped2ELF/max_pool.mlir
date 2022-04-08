//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// max_pool_1x64x16x16xfp16_2x2_pads_1x0x1x0_strides_2x2_fp16

// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIPRegMapped-to-ELF %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {

  IE.CNNNetwork entryPoint : @maxpool_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x9x8xf16, {order = #NHWC}>
  }
  func private @maxpool_f16_f16(%arg0: memref<1x64x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x64x9x8xf16, #NHWC, @DDR>) -> memref<1x64x9x8xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <9216> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>
    %2 = VPURT.DeclareBuffer "CMX_NN" [0] <9216> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 3 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
    %5 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>
    %6 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x64x16x16xf16, #NHWC, @DDR>) outputs(%0 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) updates(%4 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
    %cst = const.Declare memref<1x1x1x16xui8, #NHWC, @DDR> = dense<0> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]
    %7 = VPURT.DeclareBuffer "CMX_NN" [0] <41984> -> memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>
    %8 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%cst : memref<1x1x1x16xui8, #NHWC, @DDR>) outputs(%7 : memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>) previousDMA(%6 : !VPUIPRegMapped.Index<0>) updates(%4 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
    %cst_0 = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR> = dense<0> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %9 = VPURT.DeclareBuffer "CMX_NN" [0] <42000> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    %10 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<64x1x1x4xsi32, #NHWC, @DDR>) outputs(%9 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) previousDMA(%8 : !VPUIPRegMapped.Index<1>) updates(%4 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>
    %11 = VPUIPRegMapped.DPUInvariant {activation_window_channel_length = 16 : i32, clean_after = 0 : ui64, kernel_padding = {bottom = 1 : i64, left = 0 : i64, right = 0 : i64, top = 1 : i64}, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = "CUBOID_16x16", start_after = 0 : ui64, task_type = "MAXPOOL"} input(%0 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%9 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) parent_input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%3 : memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>) waits(%4 : !VPUIPRegMapped.Index<0>) updates(%5 : !VPUIPRegMapped.Index<1>) -> <0> PPE : {
      VPUIP.PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    }
    %12 = "VPUIPRegMapped.DPUVariant"(%11) {end = [7, 8, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 1 : i64, left = 0 : i64, right = 0 : i64, top = 1 : i64}, start = [0, 0, 0]} : (!VPUIPRegMapped.Index<0>) -> !VPUIPRegMapped.Index<0>
    %13 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%1 : memref<1x64x9x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x64x9x8xf16, #NHWC, @DDR>) previousDMA(%10 : !VPUIPRegMapped.Index<2>) waits(%5 : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<3>
    %14 = VPUIPRegMapped.MappedInference dmas(%6 : !VPUIPRegMapped.Index<0>) invariants(%11 : !VPUIPRegMapped.Index<0>) variants(%12 : !VPUIPRegMapped.Index<0>) barriers(%4 : !VPUIPRegMapped.Index<0>) dmaCount([4, 0]) invariantCount(1) variantCount(1) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPUIPRegMapped.Index<0>
    return %arg1 : memref<1x64x9x8xf16, #NHWC, @DDR>
  }
}

// CHECK: func private @maxpool_f16_f16
// CHECK: %[[VAL0:.*]] = VPURT.DeclareBuffer "CMX_NN"
// CHECK-NEXT: %[[VAL1:.*]] = VPURT.DeclareBuffer "CMX_NN"
// CHECK-NEXT: %[[VAL2:.*]] = VPURT.DeclareBuffer "CMX_NN"
// CHECK-NEXT: %[[VAL3:.*]] = VPURT.DeclareBuffer "CMX_NN"

// CHECK-NEXT: %[[VAL4:.*]] = VPUIPRegMapped.ConfigureBarrier
// CHECK-NEXT: %[[VAL5:.*]] = VPUIPRegMapped.ConfigureBarrier

// CHECK-NEXT: %[[VAL6:.*]] = VPUIPRegMapped.NNDMA
// CHECK-NEXT: %[[VALcst:.*]] = const.Declare
// CHECK-NEXT: %[[VAL7:.*]] = VPURT.DeclareBuffer "CMX_NN"

// CHECK-NEXT: %[[VAL8:.*]] = VPUIPRegMapped.NNDMA
// CHECK-NEXT: %[[VALcst_0:.*]] = const.Declare
// CHECK-NEXT: %[[VAL9:.*]] = VPURT.DeclareBuffer "CMX_NN"
// CHECK-NEXT: %[[VAL10:.*]] = VPUIPRegMapped.NNDMA

// CHECK-NEXT: %[[VAL11:.*]] = VPUIPRegMapped.DPUInvariant
// CHECK: %[[VAL12:.*]] = "VPUIPRegMapped.DPUVariant"

// CHECK-NEXT: %[[VAL13:.*]] = VPUIPRegMapped.NNDMA
// CHECK-NEXT: %[[VAL14:.*]] = VPUIPRegMapped.MappedInference

// CHECK: %[[VAL15:.*]] = ELF.CreateSection {{.*}} secName = ".text.dmaTasks0"
// CHECK-NEXT: ELF.PutOpInSection %[[VAL6]]
// CHECK-NEXT: ELF.PutOpInSection %[[VAL8]]
// CHECK-NEXT: ELF.PutOpInSection %[[VAL10]]
// CHECK-NEXT: ELF.PutOpInSection %[[VAL13]]

// CHECK: %[[VAL16:.*]] = ELF.CreateSection {{.*}} secName = ".text.BarrierConfigs"
// CHECK-NEXT: ELF.PutOpInSection %[[VAL4]]
// CHECK-NEXT: ELF.PutOpInSection %[[VAL5]]

// CHECK-DAG: %[[VAL17:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelText"
// CHECK-DAG: %[[VAL18:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelData"
// CHECK-DAG: %[[VAL19:.*]] = ELF.CreateSection {{.*}} secName = ".text.KernelParams"
// CHECK-DAG: %[[VAL20:.*]] = ELF.CreateSection {{.*}} secName = ".text.ActKernelRanges"
// CHECK-DAG: %[[VAL21:.*]] = ELF.CreateSection {{.*}} secName = ".text.ActKernelInvocations"

// CHECK: %[[VAL22:.*]] = ELF.CreateSection {{.*}} secName = ".text.MappedInference"
// CHECK-NEXT: ELF.PutOpInSection %[[VAL14]]

// CHECK: %[[VAL23:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUInvariants"
// CHECK-NEXT: ELF.PutOpInSection %[[VAL11]]

// CHECK: %[[VAL24:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUVariants"
// CHECK-NEXT: ELF.PutOpInSection %[[VAL12]]

// CHECK: %[[VAL25:.*]] = ELF.CreateMetadataSection {{.*}} secName = ".metadata"
// CHECK-NEXT: VPUIPRegMapped.NetworkMetadata

// CHECK: %[[VAL26:.*]] = ELF.Symbol  {{.*}} name("sym_dmaSection0")
// CHECK-DAG: %[[VAL27:.*]] = ELF.Symbol  {{.*}} name("sym_barrierSection")
// CHECK-DAG: %[[VAL28:.*]] = ELF.Symbol  {{.*}} name("sym_actKernelRangeSection")
// CHECK-DAG: %[[VAL29:.*]] = ELF.Symbol  {{.*}} name("sym_actKernelInvo")
// CHECK-DAG: %[[VAL30:.*]] = ELF.Symbol  {{.*}} name("sym_kernelTextSection")
// CHECK-DAG: %[[VAL31:.*]] = ELF.Symbol  {{.*}} name("sym_kernelDataSection")
// CHECK-DAG: %[[VAL32:.*]] = ELF.Symbol  {{.*}} name("sym_kernelParamsSection")
// CHECK-DAG: %[[VAL33:.*]] = ELF.Symbol  {{.*}} name("sym_inVariantsSection")
// CHECK-DAG: %[[VAL34:.*]] = ELF.Symbol  {{.*}} name("sym_variantsSection")
// CHECK-DAG: %[[VAL35:.*]] = ELF.Symbol  {{.*}} name("input_0")
// CHECK-DAG: %[[VAL36:.*]] = ELF.Symbol  {{.*}} name("output_0")

// CHECK-NEXT: %[[VAL37:.*]] = ELF.CreateSymbolTableSection secName(".symtab.input")
// CHECK-NEXT: ELF.PutOpInSection %[[VAL35]]

// CHECK: %[[VAL38:.*]] = ELF.CreateSymbolTableSection secName(".symtab.output")
// CHECK-NEXT: ELF.PutOpInSection %[[VAL36]]

// CHECK: %[[VAL39:.*]] = ELF.CreateLogicalSection {{.*}} secName = ".data.BuffersIO"

// CHECK: %[[VAL40:.*]] = ELF.CreateSection {{.*}} secName = ".data.ConstIO"
// CHECK-NEXT: ELF.PutOpInSection %cst
// CHECK-NEXT: ELF.PutOpInSection %cst_0

// CHECK-DAG: %[[VAL41:.*]] = ELF.Symbol %[[VAL40]] name("sym_constSection")
// CHECK-DAG: %[[VAL42:.*]] = ELF.Symbol %[[VAL39]] name("sym_bufferSection")

// CHECK-NEXT: %[[VAL43:.*]] = ELF.CreateSymbolTableSection secName(".symtab.buffers")
// CHECK-NEXT: ELF.PutOpInSection %[[VAL42]]
// CHECK-NEXT: ELF.PutOpInSection %[[VAL41]]

// CHECK-DAG: %[[VAL44:.*]] = ELF.Symbol %c0_i8 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR")
// CHECK-DAG: %[[VAL45:.*]] = ELF.Symbol %c1_i8 name("VPU_NNRD_SYM_RTM_IVAR")
// CHECK-DAG: %[[VAL46:.*]] = ELF.Symbol %c2_i8 name("VPU_NNRD_SYM_RTM_ACT")
// CHECK-DAG: %[[VAL47:.*]] = ELF.Symbol %c3_i8 name("VPU_NNRD_SYM_RTM_DMA0")
// CHECK-DAG: %[[VAL48:.*]] = ELF.Symbol %c4_i8 name("VPU_NNRD_SYM_RTM_DMA1")
// CHECK-DAG: %[[VAL49:.*]] = ELF.Symbol %c5_i8 name("VPU_NNRD_SYM_FIFO_BASE")
// CHECK-DAG: %[[VAL50:.*]] = ELF.Symbol %c6_i8 name("VPU_NNRD_SYM_BARRIERS_START")

// CHECK-NEXT: %[[VAL51:.*]] = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB")
// CHECK-DAG: ELF.PutOpInSection %[[VAL44]]
// CHECK-DAG: ELF.PutOpInSection %[[VAL45]]
// CHECK-DAG: ELF.PutOpInSection %[[VAL46]]
// CHECK-DAG: ELF.PutOpInSection %[[VAL47]]
// CHECK-DAG: ELF.PutOpInSection %[[VAL48]]
// CHECK-DAG: ELF.PutOpInSection %[[VAL49]]
// CHECK-DAG: ELF.PutOpInSection %[[VAL50]]

// CHECK: %[[VAL52:.*]] = ELF.CreateSymbolTableSection secName(".symtab.tasks")
// CHECK-DAG: ELF.PutOpInSection %[[VAL26]]
// CHECK-DAG: ELF.PutOpInSection %[[VAL27]]
// CHECK-DAG: ELF.PutOpInSection %[[VAL28]]
// CHECK-DAG: ELF.PutOpInSection %[[VAL29]]
// CHECK-DAG: ELF.PutOpInSection %[[VAL30]]
// CHECK-DAG: ELF.PutOpInSection %[[VAL31]]
// CHECK-DAG: ELF.PutOpInSection %[[VAL32]]
// CHECK-DAG: ELF.PutOpInSection %[[VAL33]]
// CHECK-DAG: ELF.PutOpInSection %[[VAL34]]
// CHECK-NEXT: ELF.Symbol %[[VAL14]] name("MappedInference_entry")

// CHECK: ELF.CreateRelocationSection secName(".rlt.DMA_NetInput0") sourceSymbolTableSection(%[[VAL37]]) targetSection(%[[VAL15]])
// CHECK-NEXT: ELF.RelocImmOffset baseOp(%[[VAL6]] {{.*}} %[[VAL35]]

// CHECK: ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput0") sourceSymbolTableSection(%[[VAL38]]) targetSection(%[[VAL15]])
// CHECK-NEXT: ELF.RelocImmOffset baseOp(%[[VAL13]] {{.*}} %[[VAL36]]

// CHECK: ELF.CreateRelocationSection secName(".rlt.text.dmaTasks0") sourceSymbolTableSection(%[[VAL51]]) targetSection(%[[VAL15]])
// CHECK-NEXT: ELF.Reloc baseOp(%[[VAL6]] {{.*}} offsetOf(%[[VAL0]] {{.*}} %[[VAL44]]
// CHECK-NEXT: ELF.RelocImmOffset baseOp(%[[VAL6]] {{.*}} %[[VAL47]]
// CHECK-NEXT: ELF.Reloc baseOp(%[[VAL8]] {{.*}} offsetOf(%[[VAL7]] {{.*}} %[[VAL44]]
// CHECK-NEXT: ELF.RelocImmOffset baseOp(%[[VAL8]] {{.*}} %[[VAL47]]
// CHECK-NEXT: ELF.Reloc baseOp(%[[VAL10]] {{.*}} offsetOf(%[[VAL9]] {{.*}} %[[VAL44]]
// CHECK-NEXT: ELF.RelocImmOffset baseOp(%[[VAL10]] {{.*}} %[[VAL47]]
// CHECK-NEXT: ELF.Reloc baseOp(%[[VAL13]] {{.*}} offsetOf(%[[VAL1]] {{.*}} %[[VAL44]]

// CHECK: ELF.CreateRelocationSection secName(".rlt.text.dmaTasks0") sourceSymbolTableSection(%[[VAL43]]) targetSection(%[[VAL15]])
// CHECK-NEXT: ELF.Reloc baseOp(%[[VAL8]] {{.*}} offsetOf(%cst {{.*}} %[[VAL41]]
// CHECK-NEXT: ELF.Reloc baseOp(%[[VAL10]] {{.*}} offsetOf(%cst_0 {{.*}} %[[VAL41]]

// CHECK: ELF.CreateRelocationSection secName(".rlt.text.DPUInvariants") sourceSymbolTableSection(%[[VAL51]]) targetSection(%[[VAL23]])
// CHECK-COUNT-6: ELF.RelocImmOffset baseOp(%[[VAL11]] {{.*}} %[[VAL44]]

// CHECK: ELF.CreateRelocationSection secName(".rlt.text.DPUVariants") sourceSymbolTableSection(%[[VAL51]]) targetSection(%[[VAL24]])
// CHECK-NEXT: ELF.Reloc baseOp(%[[VAL12]] {{.*}} offsetOf(%[[VAL11]] {{.*}} %[[VAL45]]
// CHECK-NEXT: ELF.RelocImmOffset baseOp(%[[VAL12]] {{.*}} %[[VAL44]]

// CHECK: ELF.CreateRelocationSection secName(".rlt.text.MappedInference") sourceSymbolTableSection(%[[VAL52]]) targetSection(%[[VAL22]])
// CHECK-DAG: ELF.Reloc baseOp(%[[VAL14]] {{.*}} offsetOf(%[[VAL6]] {{.*}} %[[VAL26]]
// CHECK-DAG: ELF.Reloc baseOp(%[[VAL14]] {{.*}} offsetOf(%[[VAL4]] {{.*}} %[[VAL27]]
// CHECK-DAG: ELF.Reloc baseOp(%[[VAL14]] {{.*}} offsetOf(%[[VAL11]] {{.*}} %[[VAL33]]
// CHECK-DAG: ELF.Reloc baseOp(%[[VAL14]] {{.*}} offsetOf(%[[VAL12]] {{.*}} %[[VAL34]]
