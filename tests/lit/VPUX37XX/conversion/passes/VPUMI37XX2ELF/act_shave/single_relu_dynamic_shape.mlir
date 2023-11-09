//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX allow-custom-values=true" --convert-VPUMI37XX-to-ELF %s | FileCheck %s

module @SimpleActivation attributes {VPU.arch = #VPU.arch_kind<VPUX37XX>, VPU.compilationMode = #VPU.compilation_mode<ReferenceSW>} {
  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096, 4096, 4096]
  module @VPU.SW {
    func.func private @builtin_ReLU(memref<2x4x20x20xf16, [@CMX_NN, 0]>, memref<2x4x20x20xf16, [@CMX_NN, 0]>) attributes {VPU.kernel_code = "relu_fp16.cpp", VPU.kernel_entry = "relu_fp16"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  IE.CNNNetwork entryPoint : @main inputsInfo : {
    DataInfo "Param_1" : tensor<2x4x20x20xf16>
    DataInfo "vpux_ie_shape_Param_1" : tensor<4xi32>
  } outputsInfo : {
    DataInfo "Relu_2" : tensor<2x4x20x20xf16>
    DataInfo "vpux_ie_shape_Relu_2" : tensor<4xi32>
  }
  func.func @main(%arg0: memref<2x4x20x20xf16, @DDR>, %arg1: memref<4xi32, @DDR>, %arg2: memref<2x4x20x20xf16, @DDR>, %arg3: memref<4xi32, @DDR>) -> (memref<2x4x20x20xf16, @DDR>, memref<4xi32, @DDR>) {
    %0 = VPURT.DeclareBuffer <NetworkInput> [0] <0> -> memref<2x4x20x20xf16, @DDR>
    %1 = VPURT.DeclareBuffer <NetworkInput> [1] <0> -> memref<4xi32, @DDR>
    %2 = VPURT.DeclareBuffer <NetworkOutput> [0] <0> -> memref<2x4x20x20xf16, @DDR>
    %3 = VPURT.DeclareBuffer <NetworkOutput> [1] <0> -> memref<4xi32, @DDR>
    %4 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2x4x20x20xf16, [@CMX_NN, 0]>
    %5 = VPURT.DeclareBuffer <CMX_NN> [0] <6400> -> memref<4xi32, [@CMX_NN, 0]>
    %6 = VPURT.DeclareBuffer <CMX_NN> [0] <6464> -> memref<2x4x20x20xf16, [@CMX_NN, 0]>
    %7 = VPURT.DeclareBuffer <CMX_NN> [0] <12864> -> memref<4xi32, [@CMX_NN, 0]>
    %8 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %10 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<2, -1> -> !VPURegMapped.Index<0:0:2>
    %11 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<3, -1> -> !VPURegMapped.Index<0:0:3>
    %12 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<4, -1> -> !VPURegMapped.Index<0:0:4>
    %13 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<2x4x20x20xf16, @DDR>) outputs(%4 : memref<2x4x20x20xf16, [@CMX_NN, 0]>) updates(%8 : !VPURegMapped.Index<0:0:0>) start_after(1) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %14 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%1 : memref<4xi32, @DDR>) outputs(%5 : memref<4xi32, [@CMX_NN, 0]>) previousDMA(%13 : !VPURegMapped.Index<0:0:0>) waits(%8 : !VPURegMapped.Index<0:0:0>) updates(%9 : !VPURegMapped.Index<0:0:1>) start_after(2) clean_after(1) -> !VPURegMapped.Index<0:0:1>
    %15 = VPUMI37XX.DeclareKernelText kernel_path("relu_fp16") -> !VPURegMapped.Index<0:0:0>
    %16 = VPUMI37XX.DeclareKernelArgs kernel_path("relu_fp16") -> !VPURegMapped.Index<0:0:0>
    %17 = VPUMI37XX.DeclareKernelEntry kernel_path("relu_fp16") -> !VPURegMapped.Index<0:0:0>
    %18 = VPUMI37XX.ActKernelRange kernel_text_index(%15 : <0:0:0>) kernel_args_index(%16 : <0:0:0>) kernel_entry_index(%17 : <0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %19 = VPUMI37XX.ActKernelInvocation range_index(%18 : <0:0:0>) waits(%9 : !VPURegMapped.Index<0:0:1>) updates(%10 : !VPURegMapped.Index<0:0:2>) tile(0) start_after(3) clean_after(2) -> !VPURegMapped.Index<0:0:0>
    %20 = VPUMI37XX.KernelParams inputs(%4 : memref<2x4x20x20xf16, [@CMX_NN, 0]>) outputs(%6 : memref<2x4x20x20xf16, [@CMX_NN, 0]>)  input_dims(%5 : memref<4xi32, [@CMX_NN, 0]>) output_dims(%7 : memref<4xi32, [@CMX_NN, 0]>) kernel_type("relu_fp16") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>
    %21 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%6 : memref<2x4x20x20xf16, [@CMX_NN, 0]>) outputs(%2 : memref<2x4x20x20xf16, @DDR>) previousDMA(%14 : !VPURegMapped.Index<0:0:1>) waits(%10 : !VPURegMapped.Index<0:0:2>) updates(%11 : !VPURegMapped.Index<0:0:3>) start_after(4) clean_after(3) -> !VPURegMapped.Index<0:0:2>
    %22 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%7 : memref<4xi32, [@CMX_NN, 0]>) outputs(%3 : memref<4xi32, @DDR>) previousDMA(%21 : !VPURegMapped.Index<0:0:2>) waits(%11 : !VPURegMapped.Index<0:0:3>) updates(%12 : !VPURegMapped.Index<0:0:4>) start_after(5) clean_after(4) -> !VPURegMapped.Index<0:0:3>
    %23 = VPUMI37XX.MappedInference dmas(%13 : !VPURegMapped.Index<0:0:0>) actKernelRanges(%18 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%19 : !VPURegMapped.Index<0:0:0>) barriers(%8 : !VPURegMapped.Index<0:0:0>) dmaCount([4, 0]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(5) -> !VPURegMapped.Index<0:0:0>
    return %arg2, %arg3 : memref<2x4x20x20xf16, @DDR>, memref<4xi32, @DDR>
  }
}

//CHECK: %[[VAL0:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL1:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL2:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL3:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL4:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL5:.*]] = VPURT.DeclareBuffer
//CHECK: %[[VAL6:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL7:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL8:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL9:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL10:.*]] = VPUMI37XX.ConfigureBarrier
//CHECK: %[[VAL11:.*]] = VPUMI37XX.NNDMA
//CHECK: %[[VAL12:.*]] = VPUMI37XX.NNDMA
//CHECK: %[[VAL13:.*]] = VPUMI37XX.DeclareKernelText
//CHECK: %[[VAL14:.*]] = VPUMI37XX.DeclareKernelArgs
//CHECK: %[[VAL15:.*]] = VPUMI37XX.DeclareKernelEntry
//CHECK: %[[VAL16:.*]] = VPUMI37XX.ActKernelRange
//CHECK: %[[VAL17:.*]] = VPUMI37XX.ActKernelInvocation
//CHECK: %[[VAL18:.*]] = VPUMI37XX.KernelParams
//CHECK: %[[VAL19:.*]] = VPUMI37XX.NNDMA
//CHECK: %[[VAL20:.*]] = VPUMI37XX.NNDMA

//CHECK: %[[BUILTIN_SYMTABSEC:.*]] = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB")
//CHECK: %[[SYMTABSEC:.*]] = ELF.CreateSymbolTableSection secName(".symtab.tasks")

//CHECK-DAG: ELF.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[SYMTABSEC]])
//CHECK-NEXT: ELF.RelocImmOffset baseOp(%[[VAL18]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-NEXT: ELF.RelocImmOffset baseOp(%[[VAL18]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>

//CHECK-DAG: ELF.CreateRelocationSection secName(".rlt.text.KernelParams") sourceSymbolTableSection(%[[BUILTIN_SYMTABSEC]])
//CHECK-NEXT: ELF.Reloc baseOp(%[[VAL18]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-NEXT: ELF.RelocImmOffset baseOp(%[[VAL18]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-NEXT: ELF.Reloc baseOp(%[[VAL18]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
//CHECK-NEXT: ELF.RelocImmOffset baseOp(%[[VAL18]] : !VPURegMapped.Index<0:0:0>) {{.*}} <R_VPU_32>
