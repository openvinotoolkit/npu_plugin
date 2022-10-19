// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --VPUIP-to-VPUIPRegMappedAndELF %s | FileCheck %s
module @Convert {
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "Parameter_6" : tensor<1x2x3x4xf16>
  } outputsInfo :  {
    DataInfo "Convert_7" : tensor<1x2x3x4xf16>
  }

  func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK:       %[[VAL1:.*]] = VPUIPRegMapped.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPUIPRegMapped.Index<0>

    %1 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x3x4xf16, @DDR>
    // CHECK:       %[[VAL2:.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x3x4xf16, @DDR>

    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %3 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    // CHECK:       %[[VAL3:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[VAL4:.*]] : memref<1x2x3x4xf16, @DDR>) outputs(%[[VAL2]] : memref<1x2x3x4xf16, @DDR>) updates(%[[VAL1]] : !VPUIPRegMapped.Index<0>) start_after(0) -> !VPUIPRegMapped.Index<0>


    %2 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x3x4xf16, @DDR>
    // CHECK:       %[[VAL5:.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x3x4xf16, @DDR>

    VPURT.Task waits(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %3 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%2 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    // CHECK:       %[[VAL6:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[VAL5]] : memref<1x2x3x4xf16, @DDR>) outputs(%[[VAL6:.*]] : memref<1x2x3x4xf16, @DDR>) previousDMA(%[[VAL3]] : !VPUIPRegMapped.Index<0>) waits(%[[VAL1]] : !VPUIPRegMapped.Index<0>) start_after(0) -> !VPUIPRegMapped.Index<1>

    // CHECK:       ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks"} -> !ELF.Section

    // CHECK:       ELF.CreateRelocationSection secName(".rlt.DMA_NetInput")
    // CHECK:           ELF.RelocImmOffset
    // CHECK:       ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput")
    // CHECK:           ELF.RelocImmOffset

    return %arg1 : memref<1x2x3x4xf16, @DDR>
    // CHECK:       return %arg1 : memref<1x2x3x4xf16, @DDR>
  }
}
