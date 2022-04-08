//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIP-to-VPUIPRegMapped %s | FileCheck %s

module @upsamplingDMA attributes {VPU.compilationMode = "DefaultHW"} {
  func @main(%arg0: memref<16x256xf16, @DDR>, %arg1: memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR> {
    %0 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<16x256xf16, @DDR>
    %1 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<16x512xf16, @DDR>
    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL0:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %4 =  VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL1:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) previousDMA(%[[VAL0]] : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %5 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 2], dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x512xf16, @DDR>) -> memref<16x512xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.UpsamplingDMAOp
    // CHECK: %[[VAL2:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x512xf16, @DDR>) previousDMA(%[[VAL1]] : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %6 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 2], dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x512xf16, @DDR>) -> memref<16x512xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.UpsamplingDMAOp
    // CHECK: %[[VAL3:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x512xf16, @DDR>) previousDMA(%[[VAL2]] : !VPUIPRegMapped.Index<2>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<3>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %7 =  VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL4:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) previousDMA(%[[VAL3]] : !VPUIPRegMapped.Index<3>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<4>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %8 =  VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL5:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) previousDMA(%[[VAL4]] : !VPUIPRegMapped.Index<4>)  start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<5>

    return %arg1 : memref<16x256xf16, @DDR>
  }
}

// -----

module @upsamplingDMA attributes {VPU.compilationMode = "DefaultHW"} {
  func @main(%arg0: memref<16x256xf16, @DDR>, %arg1: memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR> {
    %0 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<16x256xf16, @DDR>
    %1 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<16x512xf16, @DDR>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %2 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 2], dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x512xf16, @DDR>) -> memref<16x512xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.UpsamplingDMAOp
    // CHECK: %[[VAL0:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x512xf16, @DDR>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %3 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 2], dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x512xf16, @DDR>) -> memref<16x512xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.UpsamplingDMAOp
    // CHECK: %[[VAL1:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x512xf16, @DDR>) previousDMA(%[[VAL0]] : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL2:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) previousDMA(%[[VAL1]] : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %5 =  VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL3:.*]] = VPUIPRegMapped.NNDMA {port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%0 : memref<16x256xf16, @DDR>) previousDMA(%[[VAL2]] : !VPUIPRegMapped.Index<2>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<3>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %6 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 2], dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x512xf16, @DDR>) -> memref<16x512xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.UpsamplingDMAOp
    // CHECK: %[[VAL4:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x512xf16, @DDR>) previousDMA(%[[VAL3]] : !VPUIPRegMapped.Index<3>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<4>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %7 = VPUIP.UpsamplingDMAOp {upsampling_factor = [1, 2], dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x512xf16, @DDR>) -> memref<16x512xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.UpsamplingDMAOp
    // CHECK: %[[VAL5:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x512xf16, @DDR>) previousDMA(%[[VAL4]] : !VPUIPRegMapped.Index<4>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<5>

    return %arg1 : memref<16x256xf16, @DDR>
  }
}
