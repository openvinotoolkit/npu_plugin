// RUN: vpux-opt --split-input-file --canonicalize %s | FileCheck %s
module @SingleLayer attributes {VPU.arch = "VPUX30XX", VPU.compilationMode = "ReferenceSW"}  {
  module @UsedMemory  {
    IE.MemoryResource 2048 bytes of @DDR
  }
  IE.ExecutorResource {VPU.processorFrequency = 7.000000e+02 : f64} 4 of @NCE  {
    IE.ExecutorResource 5 of @DPU
  }
  IE.ExecutorResource 16 of @SHAVE_UPA
  IE.ExecutorResource 1 of @DMA_NN
  IE.MemoryResource 917504 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
  IE.MemoryResource 524288000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x1000xf16>
  } outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x1000xf16>
  }
  func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
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
      // CHECK:      ELF.Reloc 24 "R_VPU_64" %[[VAL1]] 0
      ELF.Reloc 24 "R_VPU_64" %1 0
    }
    // CHECK:    return %[[VAL2]] : memref<1x1000xf16>
    return %arg1 : memref<1x1000xf16>
  }
}
