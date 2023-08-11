//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --legalize-nd-mem-permute %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK-LABEL:   @IgnoreLegalMemPermutes
func.func @IgnoreLegalMemPermutes(%arg0: tensor<1x16x2x3xf32, {order = #NHWC}>) -> tensor<1x16x2x3xf32, {order = #NCHW}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #map} :
        tensor<1x16x2x3xf32, {order = #NHWC}> -> tensor<1x16x2x3xf32, {order = #NCHW}>
    return %0 : tensor<1x16x2x3xf32, {order = #NCHW}>

    // CHECK:     %[[VAL_0:.*]] = IE.MemPermute(%arg0) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x16x2x3xf32, {order = #NHWC}> -> tensor<1x16x2x3xf32, {order = #NCHW}>
    // CHECK:     return %[[VAL_0]] : tensor<1x16x2x3xf32, {order = #NCHW}>
}

// -----

#srcOrder = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
#dstOrder = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d7, d2, d3, d5, d6, d1, d4)>
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d7, d2, d3, d5, d6, d1, d4)>

// CHECK:     #map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4, d5, d6, d7)>
// CHECK:     #map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d7, d2, d3, d5, d6, d1, d4)>
// CHECK:     #map2 = affine_map<(d0, d1, d2, d3) -> (d1, d3, d0, d2)>

// CHECK-LABEL:   @MultipleMerge
func.func @MultipleMerge(%arg0: tensor<1x2x3x4x5x6x7x1xf32, {order = #srcOrder}>) -> tensor<1x2x3x4x5x6x7x1xf32, {order = #dstOrder}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #dstOrder, mem_perm = #map} :
        tensor<1x2x3x4x5x6x7x1xf32, {order = #srcOrder}> -> tensor<1x2x3x4x5x6x7x1xf32, {order = #dstOrder}>
    return %0 : tensor<1x2x3x4x5x6x7x1xf32, {order = #dstOrder}>

    // CHECK:     %[[CAST_1:.*]] = IE.PermuteCast(%arg0) {dst_order = #map0, mem_perm = #map0} : tensor<1x2x3x4x5x6x7x1xf32, {order = #map0}> -> tensor<1x2x3x4x5x6x7x1xf32>
    // CHECK:     %[[RESHAPE_1:.*]] = IE.Reshape(%[[CAST_1]]) {shape_value = [2, 12, 5, 42]} : tensor<1x2x3x4x5x6x7x1xf32> -> tensor<2x12x5x42xf32>
    // CHECK:     %[[PERM:.*]] = IE.MemPermute(%[[RESHAPE_1]]) {dst_order = #NCHW, mem_perm = #map2} : tensor<2x12x5x42xf32> -> tensor<12x42x2x5xf32>
    // CHECK:     %[[RESHAPE_2:.*]] = IE.Reshape(%[[PERM]]) {shape_value = [1, 1, 3, 4, 6, 7, 2, 5]} : tensor<12x42x2x5xf32> -> tensor<1x1x3x4x6x7x2x5xf32>
    // CHECK:     %[[CAST_2:.*]] = IE.PermuteCast(%[[RESHAPE_2]]) {dst_order = #map1, mem_perm = #map0} : tensor<1x1x3x4x6x7x2x5xf32> -> tensor<1x2x3x4x5x6x7x1xf32, {order = #map1}>
    // CHECK:     return %[[CAST_2]] : tensor<1x2x3x4x5x6x7x1xf32, {order = #map1}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NDHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>

// CHECK:     #map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL:   @LegalizeNDMemPermute
func.func @LegalizeNDMemPermute(%arg0: tensor<1x2x3x4x5xf32, {order = #NCDHW}>) -> tensor<1x2x3x4x5xf32, {order = #NDHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NDHWC, mem_perm = #NDHWC} :
        tensor<1x2x3x4x5xf32, {order = #NCDHW}> -> tensor<1x2x3x4x5xf32, {order = #NDHWC}>
    return %0 : tensor<1x2x3x4x5xf32, {order = #NDHWC}>

    // CHECK:     %[[CAST_1:.*]] = IE.PermuteCast(%arg0) {dst_order = #NCDHW, mem_perm = #NCDHW} : tensor<1x2x3x4x5xf32, {order = #NCDHW}> -> tensor<1x2x3x4x5xf32>
    // CHECK:     %[[RESHAPE_1:.*]] = IE.Reshape(%[[CAST_1]]) {shape_value = [2, 60, 1, 1]} : tensor<1x2x3x4x5xf32> -> tensor<2x60x1x1xf32>
    // CHECK:     %[[PERM:.*]] = IE.MemPermute(%[[RESHAPE_1]]) {dst_order = #NCHW, mem_perm = #map} : tensor<2x60x1x1xf32> -> tensor<60x2x1x1xf32>
    // CHECK:     %[[RESHAPE_2:.*]] = IE.Reshape(%[[PERM]]) {shape_value = [1, 3, 4, 5, 2]} : tensor<60x2x1x1xf32> -> tensor<1x3x4x5x2xf32>
    // CHECK:     %[[CAST_2:.*]] = IE.PermuteCast(%[[RESHAPE_2]]) {dst_order = #NDHWC, mem_perm = #NCDHW} : tensor<1x3x4x5x2xf32> -> tensor<1x2x3x4x5xf32, {order = #NDHWC}>
    // CHECK:     return %[[CAST_2]] : tensor<1x2x3x4x5xf32, {order = #NDHWC}>
}

// -----

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#NDHWC = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4, d1)>

// CHECK-LABEL:   @LegalizeNDMemPermuteNo1s
func.func @LegalizeNDMemPermuteNo1s(%arg0: tensor<2x2x3x4x5xf32, {order = #NCDHW}>) -> tensor<2x2x3x4x5xf32, {order = #NDHWC}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #NDHWC, mem_perm = #NDHWC} :
        tensor<2x2x3x4x5xf32, {order = #NCDHW}> -> tensor<2x2x3x4x5xf32, {order = #NDHWC}>
    return %0 : tensor<2x2x3x4x5xf32, {order = #NDHWC}>

    // CHECK:     %[[CAST_1:.*]] = IE.PermuteCast(%arg0) {dst_order = #NCDHW, mem_perm = #NCDHW} : tensor<2x2x3x4x5xf32, {order = #NCDHW}> -> tensor<2x2x3x4x5xf32>
    // CHECK:     %[[RESHAPE_1:.*]] = IE.Reshape(%[[CAST_1]]) {shape_value = [2, 2, 60, 1]} : tensor<2x2x3x4x5xf32> -> tensor<2x2x60x1xf32>
    // CHECK:     %[[PERM:.*]] = IE.MemPermute(%[[RESHAPE_1]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<2x2x60x1xf32> -> tensor<2x60x2x1xf32>
    // CHECK:     %[[RESHAPE_2:.*]] = IE.Reshape(%[[PERM]]) {shape_value = [2, 3, 4, 5, 2]} : tensor<2x60x2x1xf32> -> tensor<2x3x4x5x2xf32>
    // CHECK:     %[[CAST_2:.*]] = IE.PermuteCast(%[[RESHAPE_2]]) {dst_order = #NDHWC, mem_perm = #NCDHW} : tensor<2x3x4x5x2xf32> -> tensor<2x2x3x4x5xf32, {order = #NDHWC}>
    // CHECK:     return %[[CAST_2]] : tensor<2x2x3x4x5xf32, {order = #NDHWC}>
}

// -----

#SIXD = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#SIXD_PERM = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d2, d3, d1, d5)>

// CHECK:     #map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK:     #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d2, d3, d1, d5)>
// CHECK:     #map2 = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>


// CHECK-LABEL:   @LegalizeNDMemPermute6D
func.func @LegalizeNDMemPermute6D(%arg0: tensor<1x2x3x4x5x6xf32, {order = #SIXD}>) -> tensor<1x2x3x4x5x6xf32, {order = #SIXD_PERM}> {
    %0 = IE.MemPermute(%arg0) {dst_order = #SIXD_PERM, mem_perm = #SIXD_PERM} :
        tensor<1x2x3x4x5x6xf32, {order = #SIXD}> -> tensor<1x2x3x4x5x6xf32, {order = #SIXD_PERM}>
    return %0 : tensor<1x2x3x4x5x6xf32, {order = #SIXD_PERM}>

    // CHECK:     %[[CAST_1:.*]] = IE.PermuteCast(%arg0) {dst_order = #map0, mem_perm = #map0} : tensor<1x2x3x4x5x6xf32, {order = #map0}> -> tensor<1x2x3x4x5x6xf32>
    // CHECK:     %[[RESHAPE_1:.*]] = IE.Reshape(%[[CAST_1]]) {shape_value = [2, 12, 5, 6]} : tensor<1x2x3x4x5x6xf32> -> tensor<2x12x5x6xf32>
    // CHECK:     %[[PERM:.*]] = IE.MemPermute(%[[RESHAPE_1]]) {dst_order = #NCHW, mem_perm = #map2} : tensor<2x12x5x6xf32> -> tensor<5x12x2x6xf32>
    // CHECK:     %[[RESHAPE_2:.*]] = IE.Reshape(%[[PERM]]) {shape_value = [1, 5, 3, 4, 2, 6]} : tensor<5x12x2x6xf32> -> tensor<1x5x3x4x2x6xf32>
    // CHECK:     %[[CAST_2:.*]] = IE.PermuteCast(%[[RESHAPE_2]]) {dst_order = #map1, mem_perm = #map0} : tensor<1x5x3x4x2x6xf32> -> tensor<1x2x3x4x5x6xf32, {order = #map1}>
    // CHECK:     return %[[CAST_2]] : tensor<1x2x3x4x5x6xf32, {order = #map1}>
}

// -----

#CWH = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#CHW = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK-LABEL:   @LegalizeNDMemPermute3D
func.func @LegalizeNDMemPermute3D(%arg0: tensor<8x40x4096xf16, {order = #CWH}>) -> tensor<8x40x4096xf16> {
    %0 = IE.MemPermute(%arg0) {dst_order = #CHW, mem_perm = #map} :
        tensor<8x40x4096xf16, {order = #CWH}> -> tensor<8x40x4096xf16>
    return %0 : tensor<8x40x4096xf16>

    // CHECK:     %[[CAST_1:.*]] = IE.PermuteCast(%arg0) {dst_order = #CHW, mem_perm = #CHW} : tensor<8x40x4096xf16, {order = #map}> -> tensor<8x4096x40xf16>
    // CHECK:     %[[RESHAPE_1:.*]] = IE.Reshape(%[[CAST_1]]) {shape_value = [8, 4096, 40, 1]} : tensor<8x4096x40xf16> -> tensor<8x4096x40x1xf16>
    // CHECK:     %[[PERM:.*]] = IE.MemPermute(%[[RESHAPE_1]]) {dst_order = #NCHW, mem_perm = #NHCW} : tensor<8x4096x40x1xf16> -> tensor<8x40x4096x1xf16>
    // CHECK:     %[[RESHAPE_2:.*]] = IE.Reshape(%[[PERM]]) {shape_value = [8, 40, 4096]} : tensor<8x40x4096x1xf16> -> tensor<8x40x4096xf16>
    // CHECK:     %[[CAST_2:.*]] = IE.PermuteCast(%[[RESHAPE_2]]) {dst_order = #CHW, mem_perm = #CHW} : tensor<8x40x4096xf16> -> tensor<8x40x4096xf16>
    // CHECK:     return %[[CAST_2]] : tensor<8x40x4096xf16>
}
