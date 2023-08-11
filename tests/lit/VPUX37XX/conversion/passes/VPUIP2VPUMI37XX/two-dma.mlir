//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --lower-VPUIP-to-ELF %s | FileCheck %s
module @Convert {
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "Parameter_6" : tensor<1x2x3x4xf16>
  } outputsInfo :  {
    DataInfo "Convert_7" : tensor<1x2x3x4xf16>
  }

  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR> {
    %0 = VPURT.ConfigureBarrier<0> -> !VPURT.Barrier
    // CHECK:       %[[VAL1:.*]] = VPUMI37XX.ConfigureBarrier {consumer_count = 1 : ui8, producer_count = 1 : ui8}<0, -1> -> !VPURegMapped.Index<0:0:0>

    %1 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x3x4xf16, @DDR>
    // CHECK:       %[[VAL2:.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x3x4xf16, @DDR>

    VPURT.Task updates(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    // CHECK:       %[[VAL3:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%[[VAL4:.*]] : memref<1x2x3x4xf16, @DDR>) outputs(%[[VAL2]] : memref<1x2x3x4xf16, @DDR>) updates(%[[VAL1]] : !VPURegMapped.Index<0:0:0>) start_after(1) clean_after(0) -> !VPURegMapped.Index<0:0:0>


    %2 = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x3x4xf16, @DDR>
    // CHECK:       %[[VAL5:.*]] = VPURT.DeclareBuffer "DDR" <0> -> memref<1x2x3x4xf16, @DDR>

    VPURT.Task waits(%0 : !VPURT.Barrier) attributes {isTrailingSWLayer = false}  {
      %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%2 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) -> memref<1x2x3x4xf16, @DDR>
    }
    // CHECK:       %[[VAL6:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%[[VAL5]] : memref<1x2x3x4xf16, @DDR>) outputs(%[[VAL6:.*]] : memref<1x2x3x4xf16, @DDR>) previousDMA(%[[VAL3]] : !VPURegMapped.Index<0:0:0>) waits(%[[VAL1]] : !VPURegMapped.Index<0:0:0>) start_after(1) clean_after(1) -> !VPURegMapped.Index<0:0:1>

    // CHECK:       ELF.CreateSection secType(SHT_PROGBITS) secFlags("SHF_ALLOC|SHF_EXECINSTR|VPU_SHF_PROC_DMA") {secAddrAlign = 64 : i64, secInfo = 0 : i64, secName = ".text.dmaTasks0"} -> !ELF.Section

    // CHECK:       ELF.CreateRelocationSection secName(".rlt.DMA_NetInput0")
    // CHECK:           ELF.RelocImmOffset
    // CHECK:       ELF.CreateRelocationSection secName(".rlt.DMA_NetOutput0")
    // CHECK:           ELF.RelocImmOffset

    return %arg1 : memref<1x2x3x4xf16, @DDR>
    // CHECK:       return %arg1 : memref<1x2x3x4xf16, @DDR>
  }
}
