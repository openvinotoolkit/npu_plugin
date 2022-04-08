//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --unroll-per-axis-tile-dma --canonicalize %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x512x1x1xf16, #NHWC, @CMX_NN, {
    mode = "DUPLICATED",
    num_clusters =  2 : i64
}>

// CHECK-LABEL: @UnrollPerAxisTileDMAWrapInClusterDUPLICATED
func @UnrollPerAxisTileDMAWrapInClusterDUPLICATED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %0 = VPURT.DeclareBuffer "DDR" <64> -> memref<1x1x1x1xf16, #NHWC, @DDR>
    %1 = VPURT.DeclareBuffer "CMX_NN" <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
      VPUIP.NCEClusterTiling inputs(%0 as %arg0: memref<1x1x1x1xf16, #NHWC>)
              outputs(%1 as %arg1: memref<1x512x1x1xf16, #NHWC, @CMX_NN>) -> !OutputDistributed {
        VPUIP.PerAxisTileDMA {axis = 1 : i64, port = 0 : i64, tiles = 512 : i64}
              inputs(%arg0 : memref<1x1x1x1xf16, #NHWC>)
              outputs(%arg1 : memref<1x512x1x1xf16, #NHWC, @CMX_NN>) -> memref<1x512x1x1xf16, #NHWC, @CMX_NN>
      }
    }
    return %1: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT:%.*]] = VPURT.DeclareBuffer "DDR" <64> -> memref<1x1x1xf16, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x512x1x1xf16, #NHWC, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0, 1] <0> -> !VPUIP.DistributedBuffer<1x512x1xf16, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PerAxisTileDMA {dma_descriptor = {dstPlaneStride = 1024 : i64, dstStride = 1024 : i64, dstWidth = 1024 : i64, len = 1024 : i64, numPlanes = 1 : i64, srcPlaneStride = 2 : i64, srcStride = 0 : i64, srcWidth = 2 : i64}, port = 0 : i64}
    //CHECK:                inputs([[INPUT]] : memref<1x1x1xf16, @DDR>
    //CHECK:                outputs([[OUTPUT]] : !VPUIP.DistributedBuffer<1x512x1xf16, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, @CMX_NN, {mode = "DUPLICATED", num_clusters = 2 : i64}>
  
    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x16x35x16xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters =  2 : i64,
    alignment = [1, 1, 2, 1]
}>

// CHECK-LABEL: @UnrollPerAxisTileDMAWrapInClusterSEGMENTED
func @UnrollPerAxisTileDMAWrapInClusterSEGMENTED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "DDR" <64> -> memref<1x2x35x16xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer "CMX_NN" <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
        VPUIP.NCEClusterTiling inputs(%input as %arg0: memref<1x2x35x16xf16, #NHWC>)
                outputs(%output as %arg1: memref<1x16x35x16xf16, #NHWC, @CMX_NN>) -> !OutputDistributed {
            VPUIP.PerAxisTileDMA {axis = 1 : i64, port = 0 : i64, tiles = 8 : i64}
                inputs(%arg0 : memref<1x2x35x16xf16, #NHWC>)
                outputs(%arg1 : memref<1x16x35x16xf16, #NHWC, @CMX_NN>) -> memref<1x16x35x16xf16, #NHWC, @CMX_NN>
        }
    }

    return %output: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer "DDR" <1088> -> memref<32x2x1xf16, @DDR>
    //CHECK:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer "DDR" <64> -> memref<256x2x1xf16, @DDR>
    //CHECK:    [[INPUT_2:%.*]] = VPURT.DeclareBuffer "DDR" <2240> -> memref<16x2x1xf16, @DDR>
    //CHECK:    [[INPUT_3:%.*]] = VPURT.DeclareBuffer "DDR" <1216> -> memref<256x2x1xf16, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x16x35x16xf16, #NHWC, @CMX_NN, {mode = "SEGMENTED", num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64, alignment = [1, 1, 2, 1]}>
    //CHECK:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <8192> -> memref<32x16x1xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<256x16x1xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_2:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <8192> -> memref<16x16x1xf16, [@CMX_NN, 1]>
    //CHECK:    [[OUTPUT_3:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<256x16x1xf16, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PerAxisTileDMA {dma_descriptor = {dstPlaneStride = 32 : i64, dstStride = 32 : i64, dstWidth = 32 : i64, len = 32 : i64, numPlanes = 256 : i64, srcPlaneStride = 4 : i64, srcStride = 0 : i64, srcWidth = 4 : i64}, port = 0 : i64}
    //CHECK:                inputs([[INPUT_1]] : memref<256x2x1xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_1]] : memref<256x16x1xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PerAxisTileDMA {dma_descriptor = {dstPlaneStride = 32 : i64, dstStride = 32 : i64, dstWidth = 32 : i64, len = 32 : i64, numPlanes = 32 : i64, srcPlaneStride = 4 : i64, srcStride = 0 : i64, srcWidth = 4 : i64}, port = 0 : i64}
    //CHECK:                outputs([[OUTPUT_0]] : memref<32x16x1xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PerAxisTileDMA {dma_descriptor = {dstPlaneStride = 32 : i64, dstStride = 32 : i64, dstWidth = 32 : i64, len = 32 : i64, numPlanes = 256 : i64, srcPlaneStride = 4 : i64, srcStride = 0 : i64, srcWidth = 4 : i64}, port = 0 : i64}
    //CHECK:                inputs([[INPUT_3]] : memref<256x2x1xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_3]] : memref<256x16x1xf16, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PerAxisTileDMA {dma_descriptor = {dstPlaneStride = 32 : i64, dstStride = 32 : i64, dstWidth = 32 : i64, len = 32 : i64, numPlanes = 16 : i64, srcPlaneStride = 4 : i64, srcStride = 0 : i64, srcWidth = 4 : i64}, port = 0 : i64}
    //CHECK:                inputs([[INPUT_2]] : memref<16x2x1xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_2]] : memref<16x16x1xf16, [@CMX_NN, 1]>
  
    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!OutputDistributed = type !VPUIP.DistributedBuffer<
    1x4x240x240xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    kernel = [3, 3],
    pads = {bottom = 1, left = 0, right = 1, top = 0},
    strides = [2, 2],
    num_clusters = 2
}>

// CHECK-LABEL: @UnrollPerAxisTileDMAWrapInClusterOVERLAPPED
func @UnrollPerAxisTileDMAWrapInClusterOVERLAPPED() -> !OutputDistributed {
    %bar0 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    %bar1 = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier

    %input = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<1x3x240x120xf16, #NHWC, @DDR>
    %output = VPURT.DeclareBuffer "CMX_NN" <0> -> !OutputDistributed

    VPURT.Task waits(%bar0 : !VPURT.Barrier) updates(%bar1 : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
        %0 = VPUIP.NCEClusterTiling inputs(%input as %arg0: memref<1x3x240x120xf16, #NHWC>)
                outputs(%output as %arg1: memref<1x4x240x240xf16, #NHWC, @CMX_NN>) -> !OutputDistributed {
            VPUIP.PerAxisTileDMA {axis = 3 : i64, port = 0 : i64, tiles = 2 : i64}
                    inputs(%arg0 : memref<1x3x240x120xf16, #NHWC>)
                    outputs(%arg1 : memref<1x4x240x240xf16, #NHWC, @CMX_NN>) -> memref<1x4x240x240xf16, #NHWC, @CMX_NN>
        }
    }

    return %output: !OutputDistributed

    //CHECK:    [[BAR0:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[BAR1:%.*]] = VPURT.DeclareVirtualBarrier -> !VPURT.Barrier
    //CHECK:    [[INPUT_0:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <0> -> memref<121x120x4xf16, @DDR>
    //CHECK:    [[INPUT_1:%.*]] = VPURT.DeclareBuffer "NetworkInput" [0] <115200> -> memref<120x120x4xf16, @DDR>
    //CHECK:    [[OUTDISTRIBUTION:%.*]] = VPURT.DeclareBuffer "CMX_NN" <0> -> !VPUIP.DistributedBuffer<1x4x240x240xf16, #NHWC, @CMX_NN, {mode = "OVERLAPPED", num_tiles = [1, 1, 2, 1], kernel = [3, 3], pads = {bottom = 1 : i64, left = 0 : i64, right = 1 : i64, top = 0 : i64}, strides = [2, 2], num_clusters = 2 : i64}>
    //CHECK:    [[OUTPUT_0:%.*]] = VPURT.DeclareBuffer "CMX_NN" [0] <0> -> memref<121x240x4xf16, [@CMX_NN, 0]>
    //CHECK:    [[OUTPUT_1:%.*]] = VPURT.DeclareBuffer "CMX_NN" [1] <0> -> memref<120x240x4xf16, [@CMX_NN, 1]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PerAxisTileDMA {dma_descriptor = {dstPlaneStride = 1440 : i64, dstStride = 1440 : i64, dstWidth = 1440 : i64, len = 1440 : i64, numPlanes = 121 : i64, srcPlaneStride = 720 : i64, srcStride = 0 : i64, srcWidth = 720 : i64}, port = 0 : i64}
    //CHECK:                inputs([[INPUT_0]] : memref<121x120x4xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_0]] : memref<121x240x4xf16, [@CMX_NN, 0]>

    //CHECK:    VPURT.Task waits([[BAR0]] : !VPURT.Barrier) updates([[BAR1]] : !VPURT.Barrier) attributes {cycleBegin = 990 : i64, cycleEnd = 1758 : i64, isTrailingSWLayer = false} {
    //CHECK:      VPUIP.PerAxisTileDMA {dma_descriptor = {dstPlaneStride = 1440 : i64, dstStride = 1440 : i64, dstWidth = 1440 : i64, len = 1440 : i64, numPlanes = 120 : i64, srcPlaneStride = 720 : i64, srcStride = 0 : i64, srcWidth = 720 : i64}, port = 0 : i64}
    //CHECK:                inputs([[INPUT_1]] : memref<120x120x4xf16, @DDR>
    //CHECK:                outputs([[OUTPUT_1]] : memref<120x240x4xf16, [@CMX_NN, 1]>
  
    //CHECK:    return [[OUTDISTRIBUTION]] : !VPUIP.DistributedBuffer
}
