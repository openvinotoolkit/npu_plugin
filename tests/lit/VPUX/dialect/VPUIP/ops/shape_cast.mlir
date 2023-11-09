//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x3x96x160xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 3, 49, 160], [1, 3, 49, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]],
    memory_shapes = [[1, 3, 49, 160], [1, 3, 49, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x96x160xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 49, 160], [1, 16, 49, 160]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]],
    memory_shapes = [[1, 16, 49, 160], [1, 16, 49, 160]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]
}>

// CHECK-LABEL: ShapeCastWithExplicitOverlappedDistributedTensorType
func.func @ShapeCastWithExplicitOverlappedDistributedTensorType(%arg0: !InputDistributed)
    -> !OutputDistributed {

    %0 = VPUIP.ShapeCast {shape = [1, 16, 96, 160]} inputs(%arg0 : !InputDistributed) -> !OutputDistributed
    return %0 : !OutputDistributed

    // CHECK:        [[SHAPECAST:%.*]] = VPUIP.ShapeCast {shape = [1, 16, 96, 160]}
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x3x96x160xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x16x96x160xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 49, 160], [1, 16, 49, 160]], compute_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 49, 160], [1, 16, 49, 160]], memory_offsets = [[0, 0, 0, 0], [0, 0, 47, 0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed = !VPUIP.DistributedBuffer<
    1x16x88x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|MULTICASTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 44, 128], [1, 16, 44, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 44, 0]],
    memory_shapes = [[1, 16, 88, 128], [1, 16, 88, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

!OutputDistributed = !VPUIP.DistributedBuffer<
    1x16x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED|MULTICASTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 128, 128], [1, 16, 128, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]],
    memory_shapes = [[1, 16, 128, 128], [1, 16, 128, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}>

// CHECK-LABEL: ShapeCastWithExplicitSegmentedMulticastedDistributedTensorType
func.func @ShapeCastWithExplicitSegmentedMulticastedDistributedTensorType(%arg0: !InputDistributed)
    -> !OutputDistributed {

    %0 = VPUIP.ShapeCast {shape = [1, 16, 128, 128]} inputs(%arg0 : !InputDistributed) -> !OutputDistributed
    return %0 : !OutputDistributed

    // CHECK:        [[SHAPECAST:%.*]] = VPUIP.ShapeCast {shape = [1, 16, 128, 128]}
    // CHECK-SAME:         !VPUIP.DistributedBuffer<1x16x88x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:         -> !VPUIP.DistributedBuffer<1x16x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED|MULTICASTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 128, 128], [1, 16, 128, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 128, 128], [1, 16, 128, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 0, 0]]
}
