//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUMI37XX-to-ELF %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  IE.CNNNetwork entryPoint : @maxpool_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x64x16x16xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x8x8xf16, {order = #NHWC}>
  }
  func.func private @maxpool_f16_f16(%arg0: memref<1x64x16x16xf16, #NHWC, @DDR>, %arg1: memref<1x64x8x8xf16, #NHWC, @DDR>) -> memref<1x64x8x8xf16, #NHWC, @DDR> {
    %0 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 3 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %1 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>

    %cst = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR> = dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare memref<1x1x1x16xui8, #NHWC, @DDR> = dense<[[[[3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]> : tensor<1x1x1x16xui8>, [#const.Reorder<#NHWC>]

    %2 = VPURT.DeclareBuffer "CMX_NN" [0] <8192> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %3 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>
    %4 = VPURT.DeclareBuffer "CMX_NN" [0] <8192> -> memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer "CMX_NN" [0] <40960> -> memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer "CMX_NN" [0] <40976> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    %8 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x64x16x16xf16, #NHWC, @DDR>) outputs(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) updates(%0 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<1x1x1x16xui8, #NHWC, @DDR>) outputs(%6 : memref<1x1x1x16xui8, #NHWC, [@CMX_NN, 0]>) previousDMA(%8 : !VPURegMapped.Index<0:0:0>) updates(%0 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %10 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%cst : memref<64x1x1x4xsi32, #NHWC, @DDR>) outputs(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) previousDMA(%9 : !VPURegMapped.Index<0:0:1>) updates(%0 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>
    %11 = VPUMI37XX.DPUInvariant {activation_window_channel_length = 16 : i32, clean_after = 0 : ui64, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [2, 2], kernel_strides = [2, 2], mpe_frequent_mode = "CUBOID_16x16", start_after = 0 : ui64, task_type = "MAXPOOL"} input(%2 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%7 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) parent_input(%4 : memref<1x64x16x16xf16, #NHWC, [@CMX_NN, 0]>) parent_output(%5 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) waits(%0 : !VPURegMapped.Index<0:0:0>) updates(%1 : !VPURegMapped.Index<0:0:1>) -> <0:0:0> PPE : {
      VPUIP.PPETask "NOOP" {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64}
    }
    %12 = "VPUMI37XX.DPUVariant"(%11) {end = [7, 7, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]} : (!VPURegMapped.Index<0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %13 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%3 : memref<1x64x8x8xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x64x8x8xf16, #NHWC, @DDR>) previousDMA(%10 : !VPURegMapped.Index<0:0:2>) waits(%1 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:3>
    %14 = VPUMI37XX.MappedInference dmas(%8 : !VPURegMapped.Index<0:0:0>) invariants(%11 : !VPURegMapped.Index<0:0:0>) variants(%12 : !VPURegMapped.Index<0:0:0>) barriers(%0 : !VPURegMapped.Index<0:0:0>) dmaCount([4, 0]) invariantCount(1) variantCount(1) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPURegMapped.Index<0:0:0>
    return %arg1 : memref<1x64x8x8xf16, #NHWC, @DDR>
  }
}


//CHECK: %[[VAL0:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL1:.*]] = VPUMI37XX.ConfigureBarrier

//CHECK: %[[VAL8:.*]] = VPUMI37XX.NNDMA
//CHECK: %[[VAL9:.*]] = VPUMI37XX.NNDMA
//CHECK: %[[VAL10:.*]] = VPUMI37XX.NNDMA
//CHECK: %[[VAL11:.*]] = VPUMI37XX.DPUInvariant
//CHECK: %[[VAL12:.*]] = "VPUMI37XX.DPUVariant"
//CHECK: %[[VAL13:.*]] = VPUMI37XX.NNDMA

//CHECK-DAG: ELF.CreateSection {{.*}} secName = ".text.dmaTasks0"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL8]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL9]] : !VPURegMapped.Index<0:0:1>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL10]] : !VPURegMapped.Index<0:0:2>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL13]] : !VPURegMapped.Index<0:0:3>

//CHECK-DAG: ELF.CreateSection {{.*}} secName = ".text.BarrierConfigs"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL0]] : !VPURegMapped.Index<0:0:0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL1]] : !VPURegMapped.Index<0:0:1>

//CHECK-DAG: %[[IVARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUInvariants"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL11]] : !VPURegMapped.Index<0:0:0>

//CHECK-DAG: %[[VARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUVariants"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL12]] : !VPURegMapped.Index<0:0:0>

//CHECK: %[[RT_SYMTAB:.*]] = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB")

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.DPUInvariants") sourceSymbolTableSection(%[[RT_SYMTAB]]) targetSection(%[[IVARSEC]])
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32_MULTICAST_BASE"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL11]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32"

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.DPUVariants") sourceSymbolTableSection(%[[RT_SYMTAB]]) targetSection(%[[VARSEC]])
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL12]] : !VPURegMapped.Index<0:0:0>) offsetOf(%[[VAL11]] : !VPURegMapped.Index<0:0:0>) "R_VPU_32_RTM"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL12]] : !VPURegMapped.Index<0:0:0>) {{.*}} "R_VPU_32_SUM"
