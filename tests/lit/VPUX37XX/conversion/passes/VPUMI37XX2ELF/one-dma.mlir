//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --lower-VPUIP-to-ELF %s | FileCheck %s
module @OneDMA {
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x2x3x4xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x2x3x4xf16>
  }

  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    // CHECK:       %[[VAL0:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }

    // CHECK:       %[[VAL1:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks0"} -> !ELF.Section

    // CHECK:       ELF.CreateRelocationSection secName(".rlt.DMA_NetInput0")
    // CHECK:           ELF.RelocImmOffset
    // CHECK:       ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput0")
    // CHECK:           ELF.RelocImmOffset

    return %arg1 : memref<1x2x3x4xf16, @DDR>
    // CHECK:       return %arg1 : memref<1x2x3x4xf16, @DDR>
  }
}

// -----

module @DMANetworkOutputAsInput {
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x2x3x4xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x2x3x4xf16>
  }

  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    %input = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<1x2x3x4xf16, @DDR>
    // CHECK:       %[[INPUT:.*]] = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<1x2x3x4xf16, @DDR>
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {port = 0 : i64} inputs(%input : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    // CHECK:       %[[VAL0:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%[[INPUT]] : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>

    // CHECK:       ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks0"} -> !ELF.Section

    // CHECK:       %[[VAL2:.*]] = ELF.Symbol %arg0 name("input") size(48) : memref<1x2x3x4xf16, @DDR>
    // CHECK:       %[[VAL3:.*]] = ELF.Symbol %arg1 name("output") size(48) : memref<1x2x3x4xf16, @DDR>

    // CHECK:       ELF.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section
    // CHECK:           ELF.PutOpInSection %[[VAL2]]
    // CHECK:       %[[VAL4:.*]] = ELF.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section
    // CHECK:           ELF.PutOpInSection %[[VAL3]]

    // CHECK:       ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput0") sourceSymbolTableSection(%[[VAL4]])
    // CHECK:           ELF.RelocImmOffset
    // CHECK-SAME:        %[[VAL3]]
    // CHECK:           ELF.RelocImmOffset
    // CHECK-SAME:        %[[VAL3]]

    return %arg1 : memref<1x2x3x4xf16, @DDR>
    // CHECK:       return %arg1 : memref<1x2x3x4xf16, @DDR>
  }
}
