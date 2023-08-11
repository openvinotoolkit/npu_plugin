//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% allow-custom-values=true" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

module @SingleLayer attributes {VPU.compilationMode = "ReferenceSW"}  {
  module @UsedMemory  {
    IE.MemoryResource 2048 bytes of @DDR
  }
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x1000xf16>
  } outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x1000xf16>
  }
  func.func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    // CHECK:       %[[VAL0:.*]] = ELF.CreateSection secType(SHT_LOUSER) secFlags(SHF_ALLOC) {secAddrAlign = 4 : i64, secInfo = 1 : i64, secName = ".data.Weights"} -> !ELF.Section
    %0 = ELF.CreateSection secType(SHT_LOUSER) secFlags(SHF_ALLOC) {secAddrAlign = 4 : i64, secInfo = 1 : i64, secName = ".data.Weights"} -> !ELF.Section  {
    }
    // CHECK:    %[[VAL1:.*]] = ELF.Symbol %[[VAL2:.*]] name("outputCNN") : memref<1x1000xf16>
    %1 = ELF.Symbol %arg1 name("outputCNN") : memref<1x1000xf16>
    // CHECK:    %[[VAL3:.*]] = ELF.CreateSymbolTableSection secName(".output.symbolTableSection") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section
    %2 = ELF.CreateSymbolTableSection secName(".output.symbolTableSection") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section  {
      // CHECK:      ELF.PutOpInSection %[[VAL1]] : !ELF.Symbol
      ELF.PutOpInSection %1 : !ELF.Symbol
    }
    // CHECK:    %[[VAL4:.*]] = ELF.CreateRelocationSection secName(".rela.output") sourceSymbolTableSection(%[[VAL3]]) targetSection(%[[VAL0]]) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT")
    %3 = ELF.CreateRelocationSection secName(".rela.output") sourceSymbolTableSection(%2) targetSection(%0) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section  {
      // CHECK:      ELF.RelocImmOffset offset(24) "R_VPU_64" %[[VAL1]] 0
      ELF.RelocImmOffset offset(24) "R_VPU_64" %1 0
    }
    // CHECK:    return %[[VAL2]] : memref<1x1000xf16>
    return %arg1 : memref<1x1000xf16>
  }
}
