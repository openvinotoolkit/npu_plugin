// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --VPUIP-to-VPUIPRegMappedAndELF %s | FileCheck %s
module @Convert {
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x2x3x4xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x2x3x4xf16>
  }

  func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    // CHECK:       %[[VAL0:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) start_after(0) -> !VPUIPRegMapped.Index<0>
    VPURT.Task attributes {isTrailingSWLayer = false} {
      %0 = VPUIP.NNDMA {port = 0 : i64, set_crit = false, set_ord = true} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }

    // CHECK:       %[[VAL1:.*]] = ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELF.Section

    // CHECK:       ELF.CreateRelocationSection secName(".rlt.DMA_NetInput")
    // CHECK:           ELF.RelocImmOffset
    // CHECK:       ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput")
    // CHECK:           ELF.RelocImmOffset

    return %arg1 : memref<1x2x3x4xf16, @DDR>
    // CHECK:       return %arg1 : memref<1x2x3x4xf16, @DDR>
  }
}
