//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --tile-copies %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @SplitDoubleStrideCopyByChannels(
        %arg0: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
        %arg1: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
        -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
                   outputs(%arg1 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
                   -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    return %0 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_0_TILE_0:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 255, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 255, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = VPUIP.SubView %arg0 [0, 255, 0, 0] [1, 65, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = VPUIP.SubView %arg1 [0, 255, 0, 0] [1, 65, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[COPY_TILE_1:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[CONCAT:.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:      memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: return %[[CONCAT]] : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @SplitOutputDoubleStrideCopyByChannels(
        %arg0: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>,
        %arg1: memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
        -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>)
                   outputs(%arg1 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
                   -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    return %0 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_0_TILE_0:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 255, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>
    // CHECK-SAME:           to memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 255, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = VPUIP.SubView %arg0 [0, 255, 0, 0] [1, 65, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>
    // CHECK-SAME:           to memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = VPUIP.SubView %arg1 [0, 255, 0, 0] [1, 65, 32, 16] :
    // CHECK-SAME:              memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
    // CHECK-SAME:           to memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[COPY_TILE_1:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 512, 16, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: %[[CONCAT:.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x255x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:      memref<1x65x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>

    // CHECK: return %[[CONCAT]] : memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @LegalizeCopy(
        %arg0: memref<1x64x512x512xf16, #NCHW>,
        %arg1: memref<1x64x512x512xf16, #NCHW>)
        -> memref<1x64x512x512xf16, #NCHW> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x64x512x512xf16, #NCHW>)
                   outputs(%arg1 : memref<1x64x512x512xf16, #NCHW>)
                   -> memref<1x64x512x512xf16, #NCHW>

    return %0 : memref<1x64x512x512xf16, #NCHW>

    // Currently, large Copy nodes are tiled C-wise

    // Cut first tile:
    // CHECK: [[SUBVIEW_SRC_1:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_1:%.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[COPY_RET_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_1]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_1]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        -> memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Cut the second tile:
    // CHECK: [[SUBVIEW_SRC_2:%.*]] = VPUIP.SubView %arg0 [0, 31, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_2:%.*]] = VPUIP.SubView %arg1 [0, 31, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:   to memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[COPY_RET_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_2]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_2]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        -> memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // CHECK: [[SUBVIEW_SRC_3:%.*]] = VPUIP.SubView %arg0 [0, 62, 0, 0] [1, 2, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:      memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_3:%.*]] = VPUIP.SubView %arg1 [0, 62, 0, 0] [1, 2, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:      memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK: [[COPY_RET_3:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_3]] : memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_3]] : memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:          memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Concatenate the resulting output tiles:
    // CHECK: [[VAR6:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_RET_1]], [[COPY_RET_2]], [[COPY_RET_3]] :
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK-SAME:        memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x64x512x512xf16>)
    // CHECK-SAME:        -> memref<1x64x512x512xf16>
    // CHECK: return [[VAR6]] : memref<1x64x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @LegalizeStridedCopy(
        %arg0: memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>,
        %arg1: memref<1x64x512x512xf16, #NCHW>)
        -> memref<1x64x512x512xf16, #NCHW> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
                   outputs(%arg1 : memref<1x64x512x512xf16, #NCHW>)
                   -> memref<1x64x512x512xf16, #NCHW>

    return %0 : memref<1x64x512x512xf16, #NCHW>

    // Currently, large Copy nodes are tiled C-wise
    // If the Copy is strided, the strides should be preserved

    // Cut the first tile:
    // CHECK: [[SUBVIEW_SRC_1:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK-SAME:      memref<1x31x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_1:%.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 31, 512, 512] :
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:      memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // The Copy-tile preserves the original strides:
    // CHECK: [[COPY_RET_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_1]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_1]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Cut the second tile:
    // CHECK: [[SUBVIEW_SRC_2:%.*]] = VPUIP.SubView %arg0 [0, 31, 0, 0] [1, 31, 512, 512]
    // CHECK-SAME:      memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK-SAME:      memref<1x31x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_2:%.*]] = VPUIP.SubView %arg1 [0, 31, 0, 0] [1, 31, 512, 512]
    // CHECK-SAME:      memref<1x64x512x512xf16>
    // CHECK-SAME:      memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // The Copy-tile preserves the original strides:
    // CHECK: [[COPY_RET_2:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[SUBVIEW_SRC_2]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs([[SUBVIEW_DST_2]] : memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Cut the third tile:
    // CHECK: [[SUBVIEW_SRC_3:%.*]] = VPUIP.SubView %arg0 [0, 62, 0, 0] [1, 2, 512, 512]
    // CHECK-SAME:          memref<1x64x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK-SAME:          memref<1x2x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>
    // CHECK: [[SUBVIEW_DST_3:%.*]] = VPUIP.SubView %arg1 [0, 62, 0, 0] [1, 2, 512, 512]
    // CHECK-SAME:          memref<1x64x512x512xf16>
    // CHECK-SAME:          memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // CHECK: [[COPY_RET_3:%.*]] = VPUIP.Copy
    // CHECK-SAME:          inputs([[SUBVIEW_SRC_3]] : memref<1x2x512x512xf16, {order = #NCHW, strides = [33554432, 262144, 512, 1]}>)
    // CHECK-SAME:          outputs([[SUBVIEW_DST_3]] : memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:            memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>

    // Concatenate the resulting output tiles:
    // CHECK: [[RESULT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_RET_1]], [[COPY_RET_2]], [[COPY_RET_3]] :
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK-SAME:        memref<1x31x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>
    // CHECK-SAME:        memref<1x2x512x512xf16, {order = #NCHW, strides = [16777216, 262144, 512, 1]}>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x64x512x512xf16>)
    // CHECK-SAME:        memref<1x64x512x512xf16>

    // CHECK: return [[RESULT]] : memref<1x64x512x512xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func @DoNotLegalizeCopy(
        %arg0: memref<1x315x221x241xi8, #NCHW>,
        %arg1: memref<1x315x221x241xi8, #NCHW>)
        -> memref<1x315x221x241xi8, #NCHW> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x315x221x241xi8, #NCHW>)
                   outputs(%arg1 : memref<1x315x221x241xi8, #NCHW>)
                   -> memref<1x315x221x241xi8, #NCHW>

    return %0 : memref<1x315x221x241xi8, #NCHW>

    // Small enough Copy nodes (those with transaction volume less than (16MB - 1Byte)) should not be affected by the pass

    // CHECK: [[VAR0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs(%arg0 : memref<1x315x221x241xi8>)
    // CHECK-SAME:      outputs(%arg1 : memref<1x315x221x241xi8>)
    // CHECK-SAME:        -> memref<1x315x221x241xi8>
    // CHECK: return [[VAR0]] : memref<1x315x221x241xi8>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

func.func  @SplitDoubleStrideCopy(
        %arg0: memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>,
        %arg1: memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
        -> memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
                   outputs(%arg1 : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
                   -> memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    return %0 : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK: %[[ARG_0_TILE_0:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 255, 16, 1] :
    // CHECK-SAME:              memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK-SAME:           to memref<1x255x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 255, 16, 1] :
    // CHECK-SAME:              memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK-SAME:           to memref<1x255x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x255x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x255x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x255x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = VPUIP.SubView %arg0 [0, 255, 0, 0] [1, 1, 16, 1] :
    // CHECK-SAME:              memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK-SAME:           to memref<1x1x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
 
    // CHECK: %[[ARG_1_TILE_1:.*]] = VPUIP.SubView %arg1 [0, 255, 0, 0] [1, 1, 16, 1] :
    // CHECK-SAME:              memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
    // CHECK-SAME:           to memref<1x1x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
 
    // CHECK: %[[COPY_TILE_1:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x1x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x1x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x1x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
 
    // CHECK: %[[CONCAT:.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x255x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>,
    // CHECK-SAME:      memref<1x1x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>)
    // CHECK-SAME:  -> memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>

    // CHECK: return %[[CONCAT]] : memref<1x256x16x1xf16, {order = #NCHW, strides = [655360, 1024, 32, 1]}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

!SparseType =  !VPUIP.SparseBuffer<data=memref<1x320x32x16xf16, {order = #NCHW, strides = [655360, 2048, 32, 1]}>,
                                       sparsity_map=memref<1x320x32x16xi1, {order = #NCHW, strides = [655360, 2048, 32, 1]}>>

func.func @SplitByChannelsSparseBuffers(%arg0: !SparseType, %arg1: !SparseType) -> !SparseType {
    %0 = VPUIP.Copy inputs(%arg0 : !SparseType) outputs(%arg1 : !SparseType)-> !SparseType
    return %0 : !SparseType

    // CHECK:       [[ARG_0_TILE_0:%.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0] [1, 255, 32, 16] :
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x320x32x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x320x32x16xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>> to
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x255x32x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x255x32x16xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>>

    // CHECK:       [[ARG_1_TILE_0:%.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0] [1, 255, 32, 16] :
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x320x32x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x320x32x16xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>> to
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x255x32x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x255x32x16xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>>

    // CHECK:       [[COPY_TILE_0:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG_0_TILE_0]]
    // CHECK-SAME:      outputs([[ARG_1_TILE_0]]

    // CHECK:       [[ARG_0_TILE_1:%.*]] = VPUIP.SubView %arg0 [0, 255, 0, 0] [1, 65, 32, 16] :
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x320x32x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SMAE:                             sparsity_map=memref<1x320x32x16xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>> to
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x65x32x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SMAE:                             sparsity_map=memref<1x65x32x16xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>>

    // CHECK:       [[ARG_1_TILE_1:%.*]] = VPUIP.SubView %arg1 [0, 255, 0, 0] [1, 65, 32, 16] :
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x320x32x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x320x32x16xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>> to
    // CHECK-SAME:         !VPUIP.SparseBuffer<data=memref<1x65x32x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x65x32x16xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>>

    // CHECK:       [[COPY_TILE_1:%.*]] = VPUIP.Copy
    // CHECK-SAME:      inputs([[ARG_0_TILE_1]]

    // CHECK:       [[CONCAT:%.*]] = VPUIP.ConcatView
    // CHECK-SAME:      inputs([[COPY_TILE_0]], [[COPY_TILE_1]]
    // CHECK-SAME:      outputs(%arg1
    // CHECK-SAME:      -> !VPUIP.SparseBuffer<data=memref<1x320x32x16xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>,
    // CHECK-SAME:                             sparsity_map=memref<1x320x32x16xi1, {order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, strides = [655360, 2048, 32, 1]}>>

    // CHECK:       return [[CONCAT]]
}

// -----

#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

func.func @SplitByHeight3D(
        %arg0: memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>,
        %arg1: memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
        -> memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
                   outputs(%arg1 : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
                   -> memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    return %0 : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: %[[ARG_0_TILE_0:.*]] = VPUIP.SubView %arg0 [0, 0, 0] [1, 131, 64000] :
    // CHECK-SAME:              memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
    // CHECK-SAME:           to memref<1x131x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = VPUIP.SubView %arg1 [0, 0, 0] [1, 131, 64000] :
    // CHECK-SAME:              memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
    // CHECK-SAME:           to memref<1x131x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x131x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x131x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  -> memref<1x131x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = VPUIP.SubView %arg0 [0, 131, 0] [1, 129, 64000] :
    // CHECK-SAME:              memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
    // CHECK-SAME:           to memref<1x129x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = VPUIP.SubView %arg1 [0, 131, 0] [1, 129, 64000] :
    // CHECK-SAME:              memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
    // CHECK-SAME:           to memref<1x129x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: %[[COPY_TILE_1:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x129x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x129x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  -> memref<1x129x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: %[[CONCAT:.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x131x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>,
    // CHECK-SAME:      memref<1x129x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>)
    // CHECK-SAME:  -> memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>

    // CHECK: return %[[CONCAT]] : memref<1x260x64000xf16, {order = #CHW, strides = [66560000, 128000, 1]}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @SplitByChannels5D(
        %arg0: memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>,
        %arg1: memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
        -> memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
                   outputs(%arg1 : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
                   -> memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    return %0 : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: %[[ARG_0_TILE_0:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0, 0] [1, 255, 32, 16, 1] :
    // CHECK-SAME:              memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
    // CHECK-SAME:           to memref<1x255x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0, 0] [1, 255, 32, 16, 1] :
    // CHECK-SAME:              memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
    // CHECK-SAME:           to memref<1x255x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x255x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x255x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  -> memref<1x255x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = VPUIP.SubView %arg0 [0, 255, 0, 0, 0] [1, 65, 32, 16, 1] :
    // CHECK-SAME:              memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
    // CHECK-SAME:           to memref<1x65x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = VPUIP.SubView %arg1 [0, 255, 0, 0, 0] [1, 65, 32, 16, 1] :
    // CHECK-SAME:              memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
    // CHECK-SAME:           to memref<1x65x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: %[[COPY_TILE_1:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x65x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x65x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  -> memref<1x65x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: %[[CONCAT:.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x255x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>,
    // CHECK-SAME:      memref<1x65x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>)
    // CHECK-SAME:  -> memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>

    // CHECK: return %[[CONCAT]] : memref<1x320x32x16x1xf16, {order = #NCDHW, strides = [655360, 2048, 32, 1, 1]}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>

func.func @SplitByDepth5D(
        %arg0: memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>,
        %arg1: memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
        -> memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}> {
    %0 = VPUIP.Copy inputs(%arg0 : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
                   outputs(%arg1 : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
                   -> memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    return %0 : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: %[[ARG_0_TILE_0:.*]] = VPUIP.SubView %arg0 [0, 0, 0, 0, 0] [1, 1, 255, 192, 160] :
    // CHECK-SAME:              memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
    // CHECK-SAME:           to memref<1x1x255x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: %[[ARG_1_TILE_0:.*]] = VPUIP.SubView %arg1 [0, 0, 0, 0, 0] [1, 1, 255, 192, 160] :
    // CHECK-SAME:              memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
    // CHECK-SAME:           to memref<1x1x255x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: %[[COPY_TILE_0:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_0]] : memref<1x1x255x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_0]] : memref<1x1x255x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  -> memref<1x1x255x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: %[[ARG_0_TILE_1:.*]] = VPUIP.SubView %arg0 [0, 0, 255, 0, 0] [1, 1, 65, 192, 160] :
    // CHECK-SAME:              memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
    // CHECK-SAME:           to memref<1x1x65x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: %[[ARG_1_TILE_1:.*]] = VPUIP.SubView %arg1 [0, 0, 255, 0, 0] [1, 1, 65, 192, 160] :
    // CHECK-SAME:              memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
    // CHECK-SAME:           to memref<1x1x65x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: %[[COPY_TILE_1:.*]] = VPUIP.Copy
    // CHECK-SAME:  inputs(%[[ARG_0_TILE_1]] : memref<1x1x65x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  outputs(%[[ARG_1_TILE_1]] : memref<1x1x65x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  -> memref<1x1x65x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: %[[CONCAT:.*]] = VPUIP.ConcatView
    // CHECK-SAME:  inputs(%[[COPY_TILE_0]], %[[COPY_TILE_1]] :
    // CHECK-SAME:      memref<1x1x255x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>,
    // CHECK-SAME:      memref<1x1x65x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  outputs(%arg1 : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>)
    // CHECK-SAME:  -> memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>

    // CHECK: return %[[CONCAT]] : memref<1x1x320x192x160xf16, {order = #NCDHW, strides = [19660800, 19660800, 61440, 320, 1]}>
}
