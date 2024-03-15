//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --init-compiler="vpu-arch=%arch%" --mempermute-processing %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NCDHW = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map = affine_map<(d0, d1, d2, d3, d4) -> (d4, d1, d2, d3, d0)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: @MemPermuteProcessingWithNDReorder
func.func @MemPermuteProcessingWithNDReorder(%arg0: tensor<6x10x10x4x1xf16, {order = #map}>) -> tensor<6x10x10x4x1xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCDHW} : tensor<6x10x10x4x1xf16, {order = #map}> -> tensor<6x10x10x4x1xf16>
    return %0 : tensor<6x10x10x4x1xf16>

    // CHECK-NOT: IE.Reorder
    // CHECK: [[VAL0:%.*]] = IE.PermuteCast(%arg0) {dst_order = #NCDHW, mem_perm = #NCDHW} : tensor<6x10x10x4x1xf16, {order = #map}> -> tensor<1x10x10x4x6xf16>
    // CHECK: [[VAL1:%.*]] = IE.AffineReshape([[VAL0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2], [3]], shape_value = [1, 100, 4, 6]} : tensor<1x10x10x4x6xf16> -> tensor<1x100x4x6xf16>
    // CHECK: [[VAL2:%.*]] = IE.MemPermute([[VAL1]]) {dst_order = #NCHW, mem_perm = #NWCH} : tensor<1x100x4x6xf16> -> tensor<1x6x100x4xf16>
    // CHECK: [[VAL3:%.*]] = IE.AffineReshape([[VAL2]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2], [3, 4]], shape_value = [6, 10, 10, 4, 1]} : tensor<1x6x100x4xf16> -> tensor<6x10x10x4x1xf16>

    // CHECK return [[VAL3]] : tensor<6x10x10x4x1xf16>
}
