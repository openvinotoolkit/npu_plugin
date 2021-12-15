module @SingleLayer attributes {VPUIP.arch = "KMB", VPUIP.compilationMode = "ReferenceSW"}  {
  IERT.RunTimeResources availableMemory :  {
    MemoryResource 524288000 bytes of "DDR" {VPUIP.bandwidth = 8 : i64, VPUIP.derateFactor = 6.000000e-01 : f64}
    MemoryResource 917504 bytes of "CMX_NN" {VPUIP.bandwidth = 32 : i64, VPUIP.derateFactor = 1.000000e+00 : f64}
  } usedMemory :  {
    MemoryResource 2048 bytes of "DDR"
  } executors :  {
    ExecutorResource 1 of "DMA_NN"
    ExecutorResource 16 of "SHAVE_UPA"
    ExecutorResource {VPUIP.processorFrequency = 7.000000e+02 : f64} 4 of "NCE_Cluster"  {
      ExecutorResource 5 of "NCE_PerClusterDPU"
    }
  }
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "inputCNN" : tensor<1x1000xf16>
  } outputsInfo :  {
    DataInfo "outputCNN" : tensor<1x1000xf16>
  }
  func @main(%arg0: memref<1x1000xf16>, %arg1: memref<1x1000xf16>) -> memref<1x1000xf16> {
    %0 = VPUIPRegMapped.DeclareBuffer "VPU_DDR_Heap" [0] <0> -> memref<1x1000xf16, "DDR">
    %1 = VPUIPRegMapped.ConfigureBarrier<0, 1> -> !VPURT.Barrier
    %2 = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<1x1000xf16, "DDR">) outputs(%arg1 : memref<1x1000xf16>) waits(%1 : !VPURT.Barrier) start_after(0) -> memref<1x1000xf16>
    %3 = ELF.CreateSection secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 1 : i64, secName = ".data.Weights", secType = "SHT_PROGBITS"} -> !ELF.Section  {
      ELF.PutOpInSection %0 : memref<1x1000xf16, "DDR">
    }
    %4 = ELF.CreateSection secFlags(SHF_ALLOC) {secAddrAlign = 64 : i64, secInfo = 1 : i64, secName = ".data.Weights_ct", secType = "SHT_PROGBITS"} -> !ELF.Section  {
    }
    %5 = ELF.CreateSection secFlags("SHF_ALLOC|SHF_EXECINSTR") {secAddrAlign = 64 : i64, secInfo = 1 : i64, secName = ".text.dmaTasks", secType = "SHT_PROGBITS"} -> !ELF.Section  {
      ELF.PutOpInSection %2 : memref<1x1000xf16>
    }
    %6 = ELF.CreateSection secFlags(SHF_EXECINSTR) {secAddrAlign = 64 : i64, secInfo = 1 : i64, secName = ".text.BarrierConfigs", secType = "SHT_PROGBITS"} -> !ELF.Section  {
      ELF.PutOpInSection %1 : !VPURT.Barrier
    }
    %7 = ELF.Symbol %0 : memref<1x1000xf16, "DDR">
    %8 = ELF.Symbol %arg1 : memref<1x1000xf16>
    %9 = ELF.CreateSymbolTableSection secName(".rest.symbolTableSection") secFlags("SHF_NONE") -> !ELF.Section  {
      ELF.PutOpInSection %7 : !ELF.Symbol
    }
    %10 = ELF.CreateSymbolTableSection secName(".input.symbolTableSection") secFlags(VPU_SHF_USERINPUT) -> !ELF.Section  {
    }
    %11 = ELF.CreateSymbolTableSection secName(".output.symbolTableSection") secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section  {
      ELF.PutOpInSection %8 : !ELF.Symbol
    }
    %12 = ELF.CreateRelocationSection secName(".rela.dma") sourceSymbolTableSection(%9) targetSection(%5) secFlags("SHF_NONE") -> !ELF.Section  {
      ELF.Reloc 16 "R_VPU_64" %7 0
    }
    %13 = ELF.CreateRelocationSection secName(".rela.input") sourceSymbolTableSection(%10) targetSection(%5) secFlags(VPU_SHF_USERINPUT) -> !ELF.Section  {
    }
    %14 = ELF.CreateRelocationSection secName(".rela.output") sourceSymbolTableSection(%11) targetSection(%5) secFlags(VPU_SHF_USEROUTPUT) -> !ELF.Section  {
      ELF.Reloc 24 "R_VPU_64" %8 0
    }
    return %arg1 : memref<1x1000xf16>
  }
}
