// RUN: vpux-opt --split-input-file --VPUIP-to-VPUIPRegMappedAndELF %s | FileCheck %s
module @Convert attributes {VPUIP.arch = "KMB"}  {
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
    DataInfo "input" : tensor<1x2x3x4xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x2x3x4xf16>
  }

  func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    // CHECK:       %[[VAL0:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) start_after(0) -> memref<1x2x3x4xf16, @DDR>
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {port = 0 : i64, set_crit = false, set_ord = true} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }

    // CHECK:       %[[VAL1:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {secAddrAlign = 64 : i64, secInfo = 1 : i64, secName = ".text.dmaTasks"} -> !ELF.Section

    // CHECK:       ELF.CreateRelocationSection secName(".rlt.input")
    // CHECK:           ELF.Reloc
    // CHECK:       ELF.CreateRelocationSection secName(".rlt.output")
    // CHECK:           ELF.Reloc

    return %arg1 : memref<1x2x3x4xf16, @DDR>
    // CHECK:       return %arg1 : memref<1x2x3x4xf16, @DDR>
  }
}
