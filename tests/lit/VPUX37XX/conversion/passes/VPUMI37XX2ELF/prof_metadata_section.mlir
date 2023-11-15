//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=VPUX37XX" --lower-VPUIP-to-ELF %s | FileCheck %s

module @dmaSwProfiling {
  IE.CNNNetwork entryPoint : @main inputsInfo :  {
    DataInfo "input" : tensor<1x2x3x4xf16>
  } outputsInfo :  {
    DataInfo "output" : tensor<1x2x3x4xf16>
  } profilingOutputsInfo : {
    DataInfo "0_dma" : tensor<4xui32>
  }

  func.func @main(%arg0: memref<1x2x3x4xf16, @DDR>, %arg1: memref<1x2x3x4xf16, @DDR>, %arg2: memref<4xui32>) -> (memref<1x2x3x4xf16, @DDR>, memref<4xui32>) {
    %profReg = VPURT.DeclareBuffer <Register> <637702144> -> memref<1xui64, @Register>

    %profSlotStart = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<1xui64, [@CMX_NN, 0]>
    %profSlotEnd = VPURT.DeclareBuffer <CMX_NN> [0] <8> -> memref<1xui64, [@CMX_NN, 0]>

    %profBufCmx = VPURT.DeclareBuffer <CMX_NN> [0] <0> -> memref<2xui64, [@CMX_NN, 0]>
    %profOutput = VPURT.DeclareBuffer <ProfilingOutput> [0] <0> -> memref<2xui64>

    %profDmaStart = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%profReg : memref<1xui64, @Register>) outputs(%profSlotStart : memref<1xui64, [@CMX_NN, 0]>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>
    %dma = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%arg0 : memref<1x2x3x4xf16, @DDR>) outputs(%arg1 : memref<1x2x3x4xf16, @DDR>) previousDMA(%profDmaStart : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>
    %profDmaEnd = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%profReg : memref<1xui64, @Register>) outputs(%profSlotEnd : memref<1xui64, [@CMX_NN, 0]>) previousDMA(%dma : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>

    %profCmxToOut = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%profBufCmx : memref<2xui64, [@CMX_NN, 0]>) outputs(%profOutput : memref<2xui64>) previousDMA(%profDmaEnd : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:3>

    return %arg1, %arg2 : memref<1x2x3x4xf16, @DDR>, memref<4xui32>

    // CHECK: ELF.CreateProfilingSection secFlags("SHF_NONE") {secAddrAlign = 1 : i64, secInfo = 0 : i64, secName = ".profiling"}
    // CHECK-NEXT: VPUMI37XX.ProfilingMetadata
  }
}
