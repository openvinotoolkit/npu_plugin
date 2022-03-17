module @SingleLayer attributes {VPU.arch = "KMB", VPU.compilationMode = "ReferenceSW"}  {
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
    DataInfo "input" : tensor<1x1000xf16>
  } outputsInfo :  {
    DataInfo "softmax" : tensor<1x1000xf16>
  }
  func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %0 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x1000xf16, @DDR>
    %1 = VPUIPRegMapped.ConfigureBarrier<0, 1> -> !VPUIPRegMapped.Index<0>
    %2 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x1000xf16, @DDR>) outputs(%arg1 : memref<1x1000xf16>) waits(%1 : !VPUIPRegMapped.Index<0>) start_after(0) -> !VPUIPRegMapped.Index<0>
    %3 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 1 : i64, secName = ".data.Weights"} -> !ELF.Section  {
      ELF.PutOpInSection %0 : memref<1x1000xf16, @DDR>
    }
    %4 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 1 : i64, secName = ".data.Weights_ct"} -> !ELF.Section  {
    }
    %5 = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {secAddrAlign = 64 : i64, secInfo = 1 : i64, secName = ".text.dmaTasks"} -> !ELF.Section  {
      ELF.PutOpInSection %2 : !VPUIPRegMapped.Index<0>
    }
    %6 = ELF.CreateSection secType(SHT_PROGBITS) secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 1 : i64, secName = ".text.BarrierConfigs"} -> !ELF.Section  {
      ELF.PutOpInSection %1 : !VPUIPRegMapped.Index<0>
    }
    %7 = ELF.Symbol %0 name("nndmaOp0_input") : memref<1x1000xf16, @DDR>
    %8 = ELF.Symbol %arg1 name("softmax") : memref<1x1000xf16>
    %9 = ELF.CreateSymbolTableSection secName(".rest.symbolTableSection") secFlags("SHF_NONE") -> !ELF.Section  {
      ELF.PutOpInSection %7 : !ELF.Symbol
    }
    %10 = ELF.CreateSymbolTableSection secName(".symtab.input") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section  {
    }
    %11 = ELF.CreateSymbolTableSection secName(".symtab.output") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section  {
      ELF.PutOpInSection %8 : !ELF.Symbol
    }
    %12 = ELF.CreateRelocationSection secName(".rlt.dma") sourceSymbolTableSection(%9) targetSection(%5) secFlags("SHF_NONE") -> !ELF.Section  {
      ELF.Reloc 16 "R_VPU_64" %7 0
    }
    %13 = ELF.CreateRelocationSection secName(".rlt.input") sourceSymbolTableSection(%10) targetSection(%5) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USERINPUT") -> !ELF.Section  {
    }
    %14 = ELF.CreateRelocationSection secName(".rlt.output") sourceSymbolTableSection(%11) targetSection(%5) secFlags("SHF_INFO_LINK|VPU_SHF_JIT|VPU_SHF_USEROUTPUT") -> !ELF.Section  {
      ELF.Reloc 24 "R_VPU_64" %8 0
    }
    return %arg1 : memref<1x1000xf16>
  }
}

