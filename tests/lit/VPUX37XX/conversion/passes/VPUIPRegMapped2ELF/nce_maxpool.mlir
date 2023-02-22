//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIPRegMapped-to-ELF %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  IE.CNNNetwork entryPoint : @maxpool_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16, {order = #NHWC}>
  }
  func private @maxpool_f16_f16(%arg0: memref<1x64x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x64x8x8xf16, #NHWC, @DDR>) -> memref<1x64x8x8xf16, #NHWC, @DDR> {
    %0 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 3 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
    %1 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>

    %cst = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR> = dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare memref<1x1x1x16xui8, #NHWC, @DDR> = dense<[[[[3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]

    %2 = VPURT.DeclareBuffer "CMX_NN" [0] <8192> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer "CMX_NN" [0] <8192> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer "CMX_NN" [0] <40960> -> memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer "CMX_NN" [0] <40976> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    %8 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x64x16x16xf16, #NHWC, @DDR>) outputs(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) updates(%0 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
    %9 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<1x1x1x16xui8, #NHWC, @DDR>) outputs(%6 : memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>) previousDMA(%8 : !VPUIPRegMapped.Index<0>) updates(%0 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
    %10 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%cst : memref<64x1x1x4xsi32, #NHWC, @DDR>) outputs(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) previousDMA(%9 : !VPUIPRegMapped.Index<1>) updates(%0 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>
    %11 = VPUIPRegMapped.DPUInvariant {activation_window_channel_length = 16 : i32, clean_after = 0 : ui64, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = "CUBOID_16x16", start_after = 0 : ui64, task_type = "MAXPOOL"} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) parent_input(%4 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%5 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) waits(%0 : !VPUIPRegMapped.Index<0>) updates(%1 : !VPUIPRegMapped.Index<1>) -> <0> PPE : {
      VPUIP.PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    }
    %12 = "VPUIPRegMapped.DPUVariant"(%11) {end = [7, 7, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]} : (!VPUIPRegMapped.Index<0>) -> !VPUIPRegMapped.Index<0>
    %13 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x64x8x8xf16, #NHWC, @DDR>) previousDMA(%10 : !VPUIPRegMapped.Index<2>) waits(%1 : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<3>
    %14 = VPUIPRegMapped.MappedInference dmas(%8 : !VPUIPRegMapped.Index<0>) invariants(%11 : !VPUIPRegMapped.Index<0>) variants(%12 : !VPUIPRegMapped.Index<0>) barriers(%0 : !VPUIPRegMapped.Index<0>) dmaCount([4, 0]) invariantCount(1) variantCount(1) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPUIPRegMapped.Index<0>
    return %arg1 : memref<1x64x8x8xf16, #NHWC, @DDR>
  }
}


//CHECK: %[[VAL0:.*]] = VPUIPRegMapped.ConfigureBarrier
//CHECK: %[[VAL1:.*]] = VPUIPRegMapped.ConfigureBarrier

//CHECK: %[[VAL8:.*]] = VPUIPRegMapped.NNDMA
//CHECK: %[[VAL9:.*]] = VPUIPRegMapped.NNDMA
//CHECK: %[[VAL10:.*]] = VPUIPRegMapped.NNDMA
//CHECK: %[[VAL11:.*]] = VPUIPRegMapped.DPUInvariant
//CHECK: %[[VAL12:.*]] = "VPUIPRegMapped.DPUVariant"
//CHECK: %[[VAL13:.*]] = VPUIPRegMapped.NNDMA

//CHECK-DAG: ELF.CreateSection {{.*}} secName = ".text.dmaTasks0"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL8]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL9]] : !VPUIPRegMapped.Index<1>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL10]] : !VPUIPRegMapped.Index<2>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL13]] : !VPUIPRegMapped.Index<3>

//CHECK-DAG: ELF.CreateSection {{.*}} secName = ".text.BarrierConfigs"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL0]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL1]] : !VPUIPRegMapped.Index<1>

//CHECK-DAG: %[[IVARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUInvariants"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL11]] : !VPUIPRegMapped.Index<0>

//CHECK-DAG: %[[VARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUVariants"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL12]] : !VPUIPRegMapped.Index<0>

//CHECK: %[[RT_SYMTAB:.*]] = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB")

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.DPUInvariants") sourceSymbolTableSection(%[[RT_SYMTAB]]) targetSection(%[[IVARSEC]])
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32_MULTICAST_BASE"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.DPUVariants") sourceSymbolTableSection(%[[RT_SYMTAB]]) targetSection(%[[VARSEC]])
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL12]] : !VPUIPRegMapped.Index<0>) offsetOf(%[[VAL11]] : !VPUIPRegMapped.Index<0>) "R_VPU_32_RTM"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL12]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32_SUM"
