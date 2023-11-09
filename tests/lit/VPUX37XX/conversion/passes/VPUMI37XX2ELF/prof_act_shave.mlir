//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" --lower-VPUIP-to-ELF %s | FileCheck %s
module @actShaveProfiling {
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x1x1x1000xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x1x1x1000xf16>
  } profilingOutputsInfo : {
    DataInfo "0_actshave" : tensor<4xui32>
  }

  VPURT.SW.Runtime entryPoint : @VPU.SW::@runtime stack_configuration : [4096, 4096]
  module @VPU.SW {
    func.func private @builtin_hswish(memref<*xf16>, memref<*xf16>) attributes {VPU.kernel_code = "hswish_fp16.cpp", VPU.kernel_entry = "hswish_fp16"}
    func.func private @runtime() attributes {VPU.kernel_code = "nnActEntry"}
  }
  func.func @main(%arg0: memref<1x1x1x1000xf16, @DDR>, %arg1: memref<1x1x1x1000xf16, @DDR>, %arg2: memref<4xui32>) -> (memref<1x1x1x1000xf16, @DDR>, memref<4xui32>) {
    %0 = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>

    %prof_buf_cmx = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<4xui32, [@CMX_NN, 0]>
    %prof_output = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<4xui32, @DDR>

    %2 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>
    %3 = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<1, -1> -> !VPURegMapped.Index<0:0:1>
    %4 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x1x1x1000xf16, @DDR>) outputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) updates(%2 : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
 
    %5 = VPUMI37XX.DeclareKernelText kernel_path("hswish_fp16") -> !VPURegMapped.Index<0:0:0>
    %6 = VPUMI37XX.DeclareKernelArgs kernel_path("hswish_fp16") -> !VPURegMapped.Index<0:0:0>
    %7 = VPUMI37XX.DeclareKernelEntry kernel_path("hswish_fp16") -> !VPURegMapped.Index<0:0:0>
    %8 = VPUMI37XX.ActKernelRange kernel_text_index(%5 : <0:0:0>) kernel_args_index(%6 : <0:0:0>) kernel_entry_index(%7 : <0:0:0>) -> !VPURegMapped.Index<0:0:0>
    %9 = VPUMI37XX.ActKernelInvocation range_index(%8 : <0:0:0>) profiling_data(%prof_buf_cmx : memref<4xui32, [@CMX_NN, 0]>) waits(%2 : !VPURegMapped.Index<0:0:0>) updates(%3 : !VPURegMapped.Index<0:0:1>) tile(0) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %10 = VPUMI37XX.KernelParams inputs(%0 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) kernel_type("hswish_fp16") kernel_params(dense<[0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 33, 67, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]> : vector<72xui8>) -> !VPURegMapped.Index<0:0:0>
 
    %11 = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%1 : memref<1x1x1x1000xf16, [@CMX_NN, 0]>) outputs(%arg1 : memref<1x1x1x1000xf16, @DDR>) previousDMA(%4 : !VPURegMapped.Index<0:0:0>) waits(%3 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %prof_dma = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%prof_buf_cmx : memref<4xui32, [@CMX_NN, 0]>) outputs(%prof_output : memref<4xui32, @DDR>) previousDMA(%11 : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(6) -> !VPURegMapped.Index<0:0:2>


    %13 = VPUMI37XX.MappedInference dmas(%4 : !VPURegMapped.Index<0:0:0>) actKernelRanges(%8 : !VPURegMapped.Index<0:0:0>) actKernelInvocations(%9 : !VPURegMapped.Index<0:0:0>) barriers(%2 : !VPURegMapped.Index<0:0:0>) dmaCount([3]) invariantCount(0) variantCount(0) actKernelRangesCount(1) actKernelInvocationsCount(1) barrierCount(2) -> !VPURegMapped.Index<0:0:0>
    return %arg1, %arg2 : memref<1x1x1x1000xf16, @DDR>, memref<4xui32>

    // CHECK:       [[BUF_IN:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:       [[BUF_OUT:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <2000> -> memref<1x1x1x1000xf16, [@CMX_NN, 0]>
    // CHECK:       [[PROF_BUF:%.*]] = VPURT.DeclareBuffer <CMX_NN> [0] <4000> -> memref<4xui32, [@CMX_NN, 0]>
    // CHECK:       [[PROF_OUT:%.*]] = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<4xui32, @DDR>

    // CHECK:       [[BAR0:%.*]] = VPUMI37XX.ConfigureBarrier
    // CHECK:       [[BAR1:%.*]] = VPUMI37XX.ConfigureBarrier

    // CHECK:       [[DMA_IN:%.*]] = VPUMI37XX.NNDMA

    // CHECK:       [[KERNEL_TEXT:%.*]] = VPUMI37XX.DeclareKernelText
    // CHECK:       [[KERNEL_ARGS:%.*]] = VPUMI37XX.DeclareKernelArgs
    // CHECK:       [[KERNEL_ENTRY:%.*]] = VPUMI37XX.DeclareKernelEntry
    // CHECK:       [[KERNEL_RANGE:%.*]] = VPUMI37XX.ActKernelRange
    // CHECK:       [[KERNEL_INVO:%.*]] = VPUMI37XX.ActKernelInvocation
    // CHECK-SAME:       profiling_data([[PROF_BUF]] : memref<4xui32, [@CMX_NN, 0]>) 
    // CHECK:       [[KERNEL_PARAMS:%.*]] = VPUMI37XX.KernelParams

    // CHECK:       [[DMA_OUT:%.*]] = VPUMI37XX.NNDMA
    // CHECK:       [[DMA_PROF_TO_OUT:%.*]] = VPUMI37XX.NNDMA
    // CHECK-SAME:       inputs([[PROF_BUF]] : memref<4xui32, [@CMX_NN, 0]>)
    // CHECK-SAME:       outputs([[PROF_OUT]] : memref<4xui32, @DDR>) 

    // CHECK:       [[SYM_IN:%.*]] = ELF.Symbol %arg0 name("input") size(2000) : memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[SYM_OUT:%.*]] = ELF.Symbol %arg1 name("output") size(2000) : memref<1x1x1x1000xf16, @DDR>
    // CHECK:       [[SYM_PROF:%.*]] = ELF.Symbol %arg2 name("0_actshave") size(16) : memref<4xui32>


    // CHECK:       [[VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR:%.*]] = ELF.Symbol %c0_i8 name("VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR") {isBuiltin} : i8
    // CHECK:       [[VPU_NNRD_SYM_RTM_ACT:%.*]] = ELF.Symbol %c2_i8 name("VPU_NNRD_SYM_RTM_ACT") {isBuiltin} : i8
    // CHECK:       [[VPU_RT_SYMTAB:%.*]] = ELF.CreateSymbolTableSection secName("VPU_RT_SYMTAB") secFlags("SHF_NONE") {isBuiltin} -> !ELF.Section {

    // CHECK:       [[RELOC_KERNEL_INVO:%.*]] = ELF.CreateRelocationSection secName(".rlt.text.ActKernelInvocations") sourceSymbolTableSection([[VPU_RT_SYMTAB]])
    // CHECK-NEXT:       ELF.Reloc baseOp([[KERNEL_INVO]] : !VPURegMapped.Index<0:0:0>) offsetOf([[KERNEL_RANGE]] : !VPURegMapped.Index<0:0:0>) <R_VPU_32_RTM> [[VPU_NNRD_SYM_RTM_ACT]] 24
    // CHECK-NEXT:       ELF.RelocImmOffset baseOp([[KERNEL_INVO]] : !VPURegMapped.Index<0:0:0>) offset(12) <R_VPU_32_SUM> [[VPU_NNRD_SYM_NNCXM_SLICE_BASE_ADDR]] 4000
  }
}
