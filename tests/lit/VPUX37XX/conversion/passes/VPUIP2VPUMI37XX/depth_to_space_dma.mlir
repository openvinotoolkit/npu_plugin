//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --convert-VPUIP-to-VPUMI37XX %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @depthToSpace {
  func.func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, #NHWC, @DDR>) -> memref<1x1x16x256xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %3 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL0:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL1:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL0]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %5 = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.DepthToSpace
    // CHECK: %[[VAL2:.*]] = VPUMI37XX.NNDMA {dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %6 = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.DepthToSpace
    // CHECK: %[[VAL3:.*]] = VPUMI37XX.NNDMA {dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL2]] : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:3>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %7 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL4:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL3]] : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:4>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %8 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL5:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL4]] : !VPURegMapped.Index<0:0:4>)  start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:5>

    return %arg1 : memref<1x1x16x256xf16, #NHWC, @DDR>
  }
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
module @depthToSpace {
  func.func @main(%arg0: memref<1x1x16x256xf16, @DDR>, %arg1: memref<1x1x16x256xf16, #NHWC, @DDR>) -> memref<1x1x16x256xf16, #NHWC, @DDR> {
    %0 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    %1 = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %2 = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.DepthToSpace
    // CHECK: %[[VAL0:.*]] = VPUMI37XX.NNDMA {dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:0>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %3 = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.DepthToSpace
    // CHECK: %[[VAL1:.*]] = VPUMI37XX.NNDMA {dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL0]] : !VPURegMapped.Index<0:0:0>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:1>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %4 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL2:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL1]] : !VPURegMapped.Index<0:0:1>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:2>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %5 = VPUIP.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.NNDMA
    // CHECK: %[[VAL3:.*]] = VPUMI37XX.NNDMA {port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL2]] : !VPURegMapped.Index<0:0:2>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:3>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %6 = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.DepthToSpace
    // CHECK: %[[VAL4:.*]] = VPUMI37XX.NNDMA {dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL3]] : !VPURegMapped.Index<0:0:3>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:4>

    VPURT.Task attributes {cycleBegin = 0 : i64, cycleEnd = 342 : i64, isTrailingSWLayer = false} {
      %7 = VPUIP.DepthToSpaceDMA {block_size = 2 : i64, dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, mode = #IE.depth_to_space_mode<BLOCKS_FIRST>, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) -> memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>
    }

    // CHECK-NOT: VPUIP.DepthToSpace
    // CHECK: %[[VAL5:.*]] = VPUMI37XX.NNDMA {dma_descriptor = {dstPlaneStride = 3584 : i64, dstStride = 1 : i64, dstWidth = 1792 : i64, len = 1792 : i64, numPlanes = 7 : i64, srcPlaneStride = 3584 : i64, srcStride = 512 : i64, srcWidth = 256 : i64}, is_critical, port = 0 : i64} inputs(%0 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) outputs(%1 : memref<1x256x7x7xf16, #NHWC, [@CMX_NN, 0]>) previousDMA(%[[VAL4]] : !VPURegMapped.Index<0:0:4>) start_after(0) clean_after(0) -> !VPURegMapped.Index<0:0:5>

    return %arg1 : memref<1x1x16x256xf16, #NHWC, @DDR>
  }
}
