// RUN: vpux-opt --split-input-file --VPUIP-to-VPUIPRegMappedAndELF %s | FileCheck %s
module @Convert attributes {VPU.arch = "KMB"}  {
  module @UsedMemory  {
    IE.MemoryResource 48 bytes of @DDR
  }
  IE.ExecutorResource {VPU.processorFrequency = 7.000000e+02 : f64} 4 of @NCE  {
    IE.ExecutorResource 5 of @DPU 
  }
  IE.ExecutorResource 16 of @SHAVE_UPA 
  IE.ExecutorResource 1 of @DMA_NN 
  IE.MemoryResource 917504 bytes of @CMX_NN {VPU.bandwidth = 32 : i64, VPU.derateFactor = 1.000000e+00 : f64}
  IE.MemoryResource 524288000 bytes of @DDR {VPU.bandwidth = 8 : i64, VPU.derateFactor = 6.000000e-01 : f64}
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "Parameter_6" : tensor<1x2x3x4xf16>
  } outputsInfo :  {
    DataInfo "Convert_7" : tensor<1x2x3x4xf16>
  }
  func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK:       %[[VAL1:.*]] = VPUIPRegMapped.ConfigureBarrier<0, 1> -> !VPURT.Barrier

    %1 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x3x4xf16, @DDR>
    // CHECK:       %[[VAL2:.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x3x4xf16, @DDR>

    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %3 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    // CHECK:       %[[VAL3:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[VAL4:.*]] : memref<1x2x3x4xf16, @DDR>) outputs(%[[VAL2]] : memref<1x2x3x4xf16, @DDR>) updates(%[[VAL1]] : !VPURT.Barrier) start_after(0) -> memref<1x2x3x4xf16, @DDR>


    %2 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x3x4xf16, @DDR>
    // CHECK:       %[[VAL5:.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x3x4xf16, @DDR>

    VPURT.Task waits(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %3 = VPUIP.NNDMA {is_out_of_order, port = 0 : i64} inputs(%2 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    // CHECK:       %[[VAL6:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%[[VAL5]] : memref<1x2x3x4xf16, @DDR>) outputs(%[[VAL6:.*]] : memref<1x2x3x4xf16, @DDR>) waits(%[[VAL1]] : !VPURT.Barrier) start_after(0) -> memref<1x2x3x4xf16, @DDR>

    // CHECK:       ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR") {secAddrAlign = 64 : i64, secInfo = 1 : i64, secName = ".text.dmaTasks"} -> !ELF.Section

    // CHECK:       ELF.CreateRelocationSection secName(".rlt.input")
    // CHECK:           ELF.Reloc
    // CHECK:       ELF.CreateRelocationSection secName(".rlt.output")
    // CHECK:           ELF.Reloc

    return %arg1 : memref<1x2x3x4xf16, @DDR>
    // CHECK:       return %arg1 : memref<1x2x3x4xf16, @DDR>
  }
}
