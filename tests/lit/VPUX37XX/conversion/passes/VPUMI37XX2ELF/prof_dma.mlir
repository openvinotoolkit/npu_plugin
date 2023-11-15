//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" --lower-VPUIP-to-ELF %s | FileCheck %s
module @dmaSwProfiling {
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x2x3x4xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x2x3x4xf16>
  } profilingOutputsInfo : {
    DataInfo "0_dma" : tensor<4xui32>
  }

  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>, %arg2: memref<4xui32>) -> (memref<1x2x3x4xf16, @DDR>, memref<4xui32>) {

    %profReg = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register>

    %profSlotStart = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1xui64, [@CMX_NN, 0]>
    %profSlotEnd = VPURT.DeclareBuffer <CMX_NN> [0] <8> -> memref<1xui64, [@CMX_NN, 0]>

    %profBufCmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2xui64, [@CMX_NN, 0]>
    %profOutput = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<2xui64>

    %profDmaStart = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%profReg : memref<1xui64, @Register>) outputs(%profSlotStart : memref<1xui64, [@CMX_NN, 0]>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %dma = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) previousDMA(%profDmaStart : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %profDmaEnd = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%profReg : memref<1xui64, @Register>) outputs(%profSlotEnd : memref<1xui64, [@CMX_NN, 0]>) previousDMA(%dma : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>

    %profCmxToOut = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%profBufCmx : memref<2xui64, [@CMX_NN, 0]>) outputs(%profOutput : memref<2xui64>) previousDMA(%profDmaEnd : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:3>

    return %arg1, %arg2 : memref<1x2x3x4xf16, @DDR>, memref<4xui32>

    // CHECK:       [[DMA_PROF_START:%.*]] = VPUMI37XX.NNDMA
    // CHECK:       [[DMA:%.*]] = VPUMI37XX.NNDMA
    // CHECK:       [[DMA_PROF_END:%.*]] = VPUMI37XX.NNDMA
    // CHECK:       [[DMA_PROF_TO_OUT:%.*]] = VPUMI37XX.NNDMA

    // CHECK:       [[SYM_IN:%.*]] = ELF.Symbol %arg0 name("input") size(48) : memref<1x2x3x4xf16, @DDR>
    // CHECK:       [[SYM_OUT:%.*]] = ELF.Symbol %arg1 name("output") size(48) : memref<1x2x3x4xf16, @DDR>
    // CHECK:       [[SYM_PROF:%.*]] = ELF.Symbol %arg2 name("0_dma") size(16) : memref<4xui32>
    // CHECK:       [[SYMTAB_IN:%.*]] = ELF.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section {
    // CHECK-NEXT:         ELF.PutOpInSection [[SYM_IN]] : !ELF.Symbol
    // CHECK:       [[SYMTAB_OUT:%.*]] = ELF.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section {
    // CHECK-NEXT:         ELF.PutOpInSection [[SYM_OUT]] : !ELF.Symbol
    // CHECK:       [[SYMTAB_PROF:%.*]] = ELF.CreateSymbolTableSection secName(".symtab.prof_output") secFlags(VPU_SHF_PROFOUTPUT) -> !ELF.Section {
    // CHECK-NEXT:         ELF.PutOpInSection [[SYM_PROF]] : !ELF.Symbol

    // CHECK:       [[RELOCSEC_PROF:%.*]] = ELF.CreateRelocationSection secName(".rlt.DMA_ProfOutput0") sourceSymbolTableSection([[SYMTAB_PROF]])
    // CHECK-NEXT:         ELF.RelocImmOffset baseOp([[DMA_PROF_TO_OUT]] : !VPURegMapped.Index<0:0:3>) offset(24) <R_VPU_64> [[SYM_PROF]] 0

    // CHECK:       return %arg1, %arg2 : memref<1x2x3x4xf16, @DDR>, memref<4xui32>
  }
}
