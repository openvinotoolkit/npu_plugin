//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIPRegMapped-to-ELF %s | FileCheck %s
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @mainModule {
  IE.CNNNetwork entryPoint : @multiple_clusters_dpu_soh_f16_f16_f16 inputsInfo : {
    DataInfo "input_0" : tensor<1x32x32x32xf16, {order = #NHWC}>
  } outputsInfo : {
    DataInfo "output_0" : tensor<1x64x16x32xf16, {order = #NHWC}>
    DataInfo "output_1" : tensor<1x64x16x32xf16, {order = #NHWC}>
  }
  func private @multiple_clusters_dpu_soh_f16_f16_f16(%arg0: memref<1x32x32x32xf16, #NHWC, @DDR>, %arg1: memref<1x64x16x32xf16, #NHWC, @DDR>, %arg2: memref<1x64x16x32xf16, #NHWC, @DDR>) -> (memref<1x64x16x32xf16, #NHWC, @DDR>, memref<1x64x16x32xf16, #NHWC, @DDR>) {
    %0 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 4 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>
    %1 = VPUIPRegMapped.ConfigureBarrier {consumer_count = 2 : ui8, producer_count = 2 : ui8}<1, -1> -> !VPUIPRegMapped.Index<1>
    %cst = const.Declare memref<64x32x1x1xf16, #NHWC, @DDR> = dense<1.000000e+00> : tensor<64x32x1x1xf16>, [#const.Reorder<#NHWC>]
    %cst_0 = const.Declare memref<64x1x1x4xsi32, #NHWC, @DDR> = dense<1> : tensor<64x1x1x4xsi32>, [#const.Reorder<#NHWC>]
    %2 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x32x16x32xf16, #NHWC, [@DDR, 0]>
    %3 = VPURT.DeclareBuffer "NetworkInput" [0] <32768> -> memref<1x32x16x32xf16, #NHWC, [@DDR, 0]>
    %4 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %5 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>
    %7 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <4096> -> !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>
    %8 = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>
    %9 = VPURT.DeclareBuffer "CMX_NN" [1] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>
    %10 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <69632> -> !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>
    %11 = VPURT.DeclareBuffer "CMX_NN" [0] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>
    %12 = VPURT.DeclareBuffer "CMX_NN" [1] <69632> -> memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 1]>
    %13 = VPURT.DeclareBuffer "CMX_NN" [0] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>
    %14 = VPURT.DeclareBuffer "CMX_NN" [1] <102400> -> memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>
    %15 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <102400> -> !VPUIP.DistributedBuffer<64x1x1x4xsi32, {order = #NHWC, strides = [4, 1, 4, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    %16 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, [@CMX_NN, 0]>
    %17 = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, [@CMX_NN, 1]>
    %18 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%cst : memref<64x32x1x1xf16, #NHWC, @DDR>) outputs(%16, %17 : memref<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, [@CMX_NN, 0]>, memref<64x32x1x1xf16, {order = #NHWC, strides = [32, 1, 32, 32]}, [@CMX_NN, 1]>) updates(%0 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>
    %19 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%2 : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]>) outputs(%11 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%18 : !VPUIPRegMapped.Index<0>) updates(%0 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>
    %20 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%3 : memref<1x32x16x32xf16, #NHWC, [@DDR, 0]>) outputs(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 1]>) previousDMA(%19 : !VPUIPRegMapped.Index<1>) updates(%0 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>
    %21 = VPURT.DeclareBuffer "CMX_NN" [0] <102400> -> memref<64x1x1x4xsi32, {order = #NHWC, strides = [4, 1, 4, 1]}, [@CMX_NN, 0]>
    %22 = VPURT.DeclareBuffer "CMX_NN" [1] <102400> -> memref<64x1x1x4xsi32, {order = #NHWC, strides = [4, 1, 4, 1]}, [@CMX_NN, 1]>
    %23 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%cst_0 : memref<64x1x1x4xsi32, #NHWC, @DDR>) outputs(%21, %22 : memref<64x1x1x4xsi32, {order = #NHWC, strides = [4, 1, 4, 1]}, [@CMX_NN, 0]>, memref<64x1x1x4xsi32, {order = #NHWC, strides = [4, 1, 4, 1]}, [@CMX_NN, 1]>) previousDMA(%20 : !VPUIPRegMapped.Index<2>) updates(%0 : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<3>
    %24 = VPUIPRegMapped.DPUInvariant {clean_after = 0 : ui64, is_segmented, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = "CUBOID_16x16", start_after = 0 : ui64, task_type = "CONV"} input(%11 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 0]>) weights(%5 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 0]>) weight_table(%13 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 0]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%8 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) waits(%0 : !VPUIPRegMapped.Index<0>) updates(%1 : !VPUIPRegMapped.Index<1>) -> <0> PPE : {
    }
    %25 = "VPUIPRegMapped.DPUVariant"(%24) {end = [31, 15, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 0, 0]} : (!VPUIPRegMapped.Index<0>) -> !VPUIPRegMapped.Index<0>
    %26 = VPUIPRegMapped.DPUInvariant {clean_after = 0 : ui64, is_segmented, kernel_padding = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, kernel_size = [1, 1], kernel_strides = [1, 1], mpe_frequent_mode = "CUBOID_16x16", start_after = 0 : ui64, task_type = "CONV"} input(%12 : memref<1x32x16x32xf16, #NHWC, [@CMX_NN, 1]>) weights(%6 : memref<64x32x1x1xf16, #NHWC, [@CMX_NN, 1]>) weight_table(%14 : memref<64x1x1x4xsi32, #NHWC, [@CMX_NN, 1]>) parent_input(%10 : !VPUIP.DistributedBuffer<1x32x32x32xf16, {order = #NHWC, strides = [32768, 1, 1024, 32]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 1, 1]}>) parent_output(%7 : !VPUIP.DistributedBuffer<1x64x32x32xf16, {order = #NHWC, strides = [65536, 1, 2048, 64]}, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64}>) outputs(%9 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>) waits(%0 : !VPUIPRegMapped.Index<0>) updates(%1 : !VPUIPRegMapped.Index<1>) -> <1> PPE : {
    }
    %27 = "VPUIPRegMapped.DPUVariant"(%26) {end = [31, 31, 63], mpe_mode = "CUBOID_16x16", pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, start = [0, 16, 0]} : (!VPUIPRegMapped.Index<1>) -> !VPUIPRegMapped.Index<1>
    %28 = VPURT.DeclareBuffer "CMX_NN" [0] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>
    %29 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%28 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x64x16x32xf16, #NHWC, @DDR>) previousDMA(%23 : !VPUIPRegMapped.Index<3>) waits(%1 : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<4>
    %30 = VPURT.DeclareBuffer "CMX_NN" [1] <4096> -> memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>
    %31 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%30 : memref<1x64x16x32xf16, #NHWC, [@CMX_NN, 1]>) outputs(%arg2 : memref<1x64x16x32xf16, #NHWC, @DDR>) previousDMA(%29 : !VPUIPRegMapped.Index<4>) waits(%1 : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<5>
    %32 = VPUIPRegMapped.MappedInference dmas(%18 : !VPUIPRegMapped.Index<0>) invariants(%24 : !VPUIPRegMapped.Index<0>) variants(%25 : !VPUIPRegMapped.Index<0>) barriers(%0 : !VPUIPRegMapped.Index<0>) dmaCount([6, 0]) invariantCount(2) variantCount(2) actKernelRangesCount(0) actKernelInvocationsCount(0) barrierCount(2) -> !VPUIPRegMapped.Index<0>

    return %arg1, %arg2 : memref<1x64x16x32xf16, #NHWC, @DDR>, memref<1x64x16x32xf16, #NHWC, @DDR>
  }
}

//CHECK-LABEL: @multiple_clusters_dpu_soh_f16_f16_f16
//CHECK: %[[VAL0:.*]] = VPUIPRegMapped.ConfigureBarrier
//CHECK: %[[VAL1:.*]] = VPUIPRegMapped.ConfigureBarrier
//CHECK: %[[VAL18:.*]] = VPUIPRegMapped.NNDMA
//CHECK: %[[VAL19:.*]] = VPUIPRegMapped.NNDMA
//CHECK: %[[VAL20:.*]] = VPUIPRegMapped.NNDMA
//CHECK: %[[VAL23:.*]] = VPUIPRegMapped.NNDMA
//CHECK: %[[VAL24:.*]] = VPUIPRegMapped.DPUInvariant
//CHECK: %[[VAL25:.*]] = "VPUIPRegMapped.DPUVariant"
//CHECK: %[[VAL26:.*]] = VPUIPRegMapped.DPUInvariant
//CHECK: %[[VAL27:.*]] = "VPUIPRegMapped.DPUVariant"
//CHECK: %[[VAL29:.*]] = VPUIPRegMapped.NNDMA
//CHECK: %[[VAL31:.*]] = VPUIPRegMapped.NNDMA

//CHECK-DAG: %[[DMASEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.dmaTasks0"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL18]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL19]] : !VPUIPRegMapped.Index<1>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL20]] : !VPUIPRegMapped.Index<2>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL23]] : !VPUIPRegMapped.Index<3>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL29]] : !VPUIPRegMapped.Index<4>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL31]] : !VPUIPRegMapped.Index<5>

//CHECK-DAG: %[[BARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.BarrierConfigs"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL0]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL1]] : !VPUIPRegMapped.Index<1>

//CHECK-DAG: %[[INVSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUInvariants"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL24]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL26]] : !VPUIPRegMapped.Index<1>

//CHECK-DAG: %[[VARSEC:.*]] = ELF.CreateSection {{.*}} secName = ".text.DPUVariants"
//CHECK-NEXT: ELF.PutOpInSection %[[VAL25]] : !VPUIPRegMapped.Index<0>
//CHECK-NEXT: ELF.PutOpInSection %[[VAL27]] : !VPUIPRegMapped.Index<1>

//CHECK-DAG: ELF.CreateMetadataSection {{.*}} secName = ".metadata"
//CHECK-NEXT: VPUIPRegMapped.NetworkMetadata

//CHECK: %[[BUILTIN_SYMTABSEC:.*]] = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB")
//CHECK: %[[SYMTABSEC:.*]] = ELF.CreateSymbolTableSection secName(".symtab.tasks")

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.dmaTasks0") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL18]] : !VPUIPRegMapped.Index<0>)
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL19]] : !VPUIPRegMapped.Index<1>)
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL20]] : !VPUIPRegMapped.Index<2>)
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL23]] : !VPUIPRegMapped.Index<3>)
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL29]] : !VPUIPRegMapped.Index<4>)
//CHECK-DAG: ELF.Reloc baseOp(%[[VAL31]] : !VPUIPRegMapped.Index<5>)
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL18]] : !VPUIPRegMapped.Index<0>) offset(0) "R_VPU_32_RTM"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL19]] : !VPUIPRegMapped.Index<1>) offset(0) "R_VPU_32_RTM"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL20]] : !VPUIPRegMapped.Index<2>) offset(0) "R_VPU_32_RTM"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL23]] : !VPUIPRegMapped.Index<3>) offset(0) "R_VPU_32_RTM"
//CHECK-DAG: ELF.RelocImmOffset baseOp(%[[VAL29]] : !VPUIPRegMapped.Index<4>) offset(0) "R_VPU_32_RTM"

//CHECK: ELF.CreateRelocationSection secName(".rlt.text.DPUInvariants") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL24]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL24]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL24]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL24]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL24]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL24]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32_MULTICAST_BASE"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL24]] : !VPUIPRegMapped.Index<0>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL28]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL28]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL28]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL28]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL28]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL28]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32_MULTICAST_BASE"
//CHECK-DAG ELF.RelocImmOffset baseOp(%[[VAL28]] : !VPUIPRegMapped.Index<1>) {{.*}} "R_VPU_32"
