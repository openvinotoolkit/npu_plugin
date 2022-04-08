//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-VPUIP-to-VPUIPRegMapped %s | FileCheck %s

module @perAxisTileDMA attributes {VPU.compilationMode = "DefaultHW"} {
  func @main(%arg0: memref<16x256xf16, @DDR>, %arg1: memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR> {
    %0 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<16x256xf16, @DDR>
    %1 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<16x256xf16, @DDR>
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
      %5 = VPUIP.PerAxisTileDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.PerAxisTileDMA
    // CHECK: %[[VAL2:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x256xf16, @DDR>) previousDMA(%[[VAL1]] : !VPUIPRegMapped.Index<1>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<2>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %6 = VPUIP.PerAxisTileDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.PerAxisTileDMA
    // CHECK: %[[VAL3:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x256xf16, @DDR>) previousDMA(%[[VAL2]] : !VPUIPRegMapped.Index<2>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<3>

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

module @perAxisTileDMA attributes {VPU.compilationMode = "DefaultHW"} {
  func @main(%arg0: memref<16x256xf16, @DDR>, %arg1: memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR> {
    %0 = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<16x256xf16, @DDR>
    %1 = VPURT.DeclareBuffer "NetworkOutput" [0] <0> -> memref<16x256xf16, @DDR>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %2 = VPUIP.PerAxisTileDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.PerAxisTileDMA
    // CHECK: %[[VAL0:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x256xf16, @DDR>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %3 = VPUIP.PerAxisTileDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.PerAxisTileDMA
    // CHECK: %[[VAL1:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x256xf16, @DDR>) previousDMA(%[[VAL0]] : !VPUIPRegMapped.Index<0>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<1>

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
      %6 = VPUIP.PerAxisTileDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.PerAxisTileDMA
    // CHECK: %[[VAL4:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x256xf16, @DDR>) previousDMA(%[[VAL3]] : !VPUIPRegMapped.Index<3>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<4>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %7 = VPUIP.PerAxisTileDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x256xf16, @DDR>) -> memref<16x256xf16, @DDR>
    }

    // CHECK-NOT: VPUIP.PerAxisTileDMA
    // CHECK: %[[VAL5:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<16x256xf16, @DDR>) outputs(%1 : memref<16x256xf16, @DDR>) previousDMA(%[[VAL4]] : !VPUIPRegMapped.Index<4>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<5>

    return %arg1 : memref<16x256xf16, @DDR>
  }
}

// -----

#NC = affine_map<(d0, d1) -> (d0, d1)>
module @perAxisTileDMA {
func @UnrollDistributedPerAxisTileDMAOutput(%arg0: memref<1x16x16x16xf16, @DDR>, %arg1: memref<64x32x1x1xf16, @DDR>) -> memref<64x32x1x1xf16, @DDR> {
  %cst = const.Declare memref<16x256xf16, #NC, @DDR> = dense<1.000000e+00> : tensor<16x256xf16>, [#const.Reorder<#NC>]
  // CHECK: %[[CST:.*]] = const.Declare memref<16x256xf16, @DDR>

  %3 = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<16x256xf16, {order = #NC, strides = [256, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  // CHECK: %[[WEIGHTS:.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<16x256xf16, {order = #NC, strides = [256, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

  VPURT.Task attributes {cycleBegin = 1410 : i64, cycleEnd = 1613 : i64, isTrailingSWLayer = false} {
    %4 = VPUIP.PerAxisTileDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, port = 0 : i64} inputs(%cst : memref<16x256xf16, #NC, @DDR>) outputs(%3 : !VPUIP.DistributedBuffer<16x256xf16, {order = #NC, strides = [256, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>) -> !VPUIP.DistributedBuffer<16x256xf16, {order = #NC, strides = [256, 1]}, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  }
  // CHECK: %[[BUFF_TILE_0:.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<16x256xf16, [@CMX_NN, 0]>
  // CHECK: %[[BUFF_TILE_1:.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<16x256xf16, [@CMX_NN, 1]>
  // CHECK-NOT: VPURT.Task
  // CHECK: %[[DMA0:.*]] = VPUIPRegMapped.NNDMA {dma_descriptor = {dstPlaneStride = 2 : i64, dstStride = 32 : i64, dstWidth = 2 : i64, len = 512 : i64, numPlanes = 16 : i64, srcPlaneStride = 512 : i64, srcStride = 2 : i64, srcWidth = 512 : i64}, is_critical, port = 0 : i64} inputs(%[[CST]] : memref<16x256xf16, @DDR>) outputs(%[[BUFF_TILE_0]], %[[BUFF_TILE_1]] : memref<16x256xf16, [@CMX_NN, 0]>, memref<16x256xf16, [@CMX_NN, 1]>) start_after(0) clean_after(0) -> !VPUIPRegMapped.Index<0>

  return %arg1 : memref<64x32x1x1xf16, @DDR>
}
}
