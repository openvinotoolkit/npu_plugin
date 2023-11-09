//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// -----

// CHECK-LABEL: @OneInputFold
func.func @OneInputFold(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = VPU.Concat(%arg0) { per_axis = #IE.Concat<axis = 1> } : tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>

    // CHECK-NOT: VPU.Concat
    // CHECK:     return %arg0
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: FuseRightConcat
func.func @FuseRightConcat(%arg0: tensor<1x64x250x250xf16, {order = #NHWC}>,
                           %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>,
                           %arg2: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> tensor<1x96x250x250xf16, {order = #NHWC}> {

    %TILING = VPU.Concat(%arg1, %arg2) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %MAIN_CONCAT = VPU.Concat(%arg0, %TILING) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 64, 0, 0]
        ]
    } : tensor<1x64x250x250xf16, {order = #NHWC}>,
        tensor<1x32x250x250xf16, {order = #NHWC}>
            -> tensor<1x96x250x250xf16, {order = #NHWC}>

    return %MAIN_CONCAT : tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   [[MAIN_CONCAT:%.*]] = VPU.Concat(%arg0, %arg1, %arg2) {
    // CHECK-SAME:       static_offsets = [
    // CHECK-SAME:           [0, 0, 0, 0],
    // CHECK-SAME:           [0, 64, 0, 0],
    // CHECK-SAME:           [0, 64, 125, 0]
    // CHECK-SAME:       ]
    // CHECK-SAME:   } : tensor<1x64x250x250xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x32x125x250xf16, {order = #NHWC}>
    // CHECK-SAME:           -> tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   return [[MAIN_CONCAT]] : tensor<1x96x250x250xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: FuseLeftConcat
func.func @FuseLeftConcat(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                          %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>,
                          %arg2: tensor<1x64x250x250xf16, {order = #NHWC}>)
    -> tensor<1x96x250x250xf16, {order = #NHWC}> {

    %TILING = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %MAIN_CONCAT = VPU.Concat(%TILING, %arg2) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 32, 0, 0]
        ]
    } : tensor<1x32x250x250xf16, {order = #NHWC}>,
        tensor<1x64x250x250xf16, {order = #NHWC}>
            -> tensor<1x96x250x250xf16, {order = #NHWC}>

    return %MAIN_CONCAT : tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   [[MAIN_CONCAT:%.*]] = VPU.Concat(%arg0, %arg1, %arg2) {
    // CHECK-SAME:       static_offsets = [
    // CHECK-SAME:           [0, 0, 0, 0],
    // CHECK-SAME:           [0, 0, 125, 0],
    // CHECK-SAME:           [0, 32, 0, 0]
    // CHECK-SAME:       ]
    // CHECK-SAME:   } : tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x64x250x250xf16, {order = #NHWC}>
    // CHECK-SAME:           -> tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   return [[MAIN_CONCAT]] : tensor<1x96x250x250xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: FuseTwoConcats
func.func @FuseTwoConcats(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                          %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>,
                          %arg2: tensor<1x64x250x125xf16, {order = #NHWC}>,
                          %arg3: tensor<1x64x250x125xf16, {order = #NHWC}>)
    -> tensor<1x96x250x250xf16, {order = #NHWC}> {

    %LHS_TILING = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %RHS_TILING = VPU.Concat(%arg2, %arg3) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 0, 125]
        ]
    } : tensor<1x64x250x125xf16, {order = #NHWC}>,
        tensor<1x64x250x125xf16, {order = #NHWC}>
            -> tensor<1x64x250x250xf16, {order = #NHWC}>

    %MAIN_CONCAT = VPU.Concat(%LHS_TILING, %RHS_TILING) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 32, 0, 0]
        ]
    } : tensor<1x32x250x250xf16, {order = #NHWC}>,
        tensor<1x64x250x250xf16, {order = #NHWC}>
            -> tensor<1x96x250x250xf16, {order = #NHWC}>

    return %MAIN_CONCAT : tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   [[MAIN_CONCAT:%.*]] = VPU.Concat(%arg0, %arg1, %arg2, %arg3) {
    // CHECK-SAME:       static_offsets = [
    // CHECK-SAME:           [0, 0, 0, 0],
    // CHECK-SAME:           [0, 0, 125, 0],
    // CHECK-SAME:           [0, 32, 0, 0]
    // CHECK-SAME:           [0, 32, 0, 125]
    // CHECK-SAME:       ]
    // CHECK-SAME:   } : tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x64x250x125xf16, {order = #NHWC}>,
    // CHECK-SAME:       tensor<1x64x250x125xf16, {order = #NHWC}>
    // CHECK-SAME:           -> tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   return [[MAIN_CONCAT]] : tensor<1x96x250x250xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: SkipConcatWithTwoConsumers
func.func @SkipConcatWithTwoConsumers(%arg0: tensor<1x64x250x250xf16, {order = #NHWC}>,
                                      %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>,
                                      %arg2: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x96x250x250xf16, {order = #NHWC}>, tensor<1x32x250x250xf16, {order = #NHWC}>)  {

    %TILING = VPU.Concat(%arg1, %arg2) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %MAIN_CONCAT = VPU.Concat(%arg0, %TILING) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 64, 0, 0]
        ]
    } : tensor<1x64x250x250xf16, {order = #NHWC}>,
        tensor<1x32x250x250xf16, {order = #NHWC}>
            -> tensor<1x96x250x250xf16, {order = #NHWC}>

    return %MAIN_CONCAT, %TILING :
        tensor<1x96x250x250xf16, {order = #NHWC}>,
        tensor<1x32x250x250xf16, {order = #NHWC}>

    // CHECK:   [[TILING_CONCAT:%.*]] = VPU.Concat(%arg1, %arg2)
    // CHECK:   [[MAIN_CONCAT:%.*]] = VPU.Concat(%arg0, [[TILING_CONCAT]])

    // CHECK:   return [[MAIN_CONCAT]], [[TILING_CONCAT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: FuseConcatWithTwoConsumers
func.func @FuseConcatWithTwoConsumers(%arg0: tensor<1x64x250x250xf16, {order = #NHWC}>,
                                      %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>,
                                      %arg2: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> (tensor<1x96x250x250xf16, {order = #NHWC}>, tensor<1x96x250x250xf16, {order = #NHWC}>) {

    %TILING = VPU.Concat(%arg1, %arg2) {
        per_axis = #IE.Concat<axis = 2>
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %MAIN_CONCAT = VPU.Concat(%arg0, %TILING) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 64, 0, 0]
        ]
    } : tensor<1x64x250x250xf16, {order = #NHWC}>,
        tensor<1x32x250x250xf16, {order = #NHWC}>
            -> tensor<1x96x250x250xf16, {order = #NHWC}>

    %ANOTHER_CONCAT = VPU.Concat(%TILING, %arg0) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 32, 0, 0]
        ]
    } : tensor<1x32x250x250xf16, {order = #NHWC}>,
        tensor<1x64x250x250xf16, {order = #NHWC}>
            -> tensor<1x96x250x250xf16, {order = #NHWC}>

    return %MAIN_CONCAT, %ANOTHER_CONCAT :
                tensor<1x96x250x250xf16, {order = #NHWC}>,
                tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   [[MAIN_CONCAT:%.*]] = VPU.Concat(%arg0, %arg1, %arg2) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 64, 0, 0],
    // CHECK-SAME:          [0, 64, 125, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  } : tensor<1x64x250x250xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x32x125x250xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   [[ANOTHER_CONCAT:%.*]] = VPU.Concat(%arg1, %arg2, %arg0) {
    // CHECK-SAME:      static_offsets = [
    // CHECK-SAME:          [0, 0, 0, 0],
    // CHECK-SAME:          [0, 0, 125, 0],
    // CHECK-SAME:          [0, 32, 0, 0]
    // CHECK-SAME:      ]
    // CHECK-SAME:  } : tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x32x125x250xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<1x64x250x250xf16, {order = #NHWC}>
    // CHECK-SAME:          -> tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   return %0, %1 : tensor<1x96x250x250xf16, {order = #NHWC}>, tensor<1x96x250x250xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: SkipConstants
func.func @SkipConstants(%arg0: tensor<1x32x125x250xf16, {order = #NHWC}>,
                         %arg1: tensor<1x32x125x250xf16, {order = #NHWC}>)
    -> tensor<1x96x250x250xf16, {order = #NHWC}> {

    %CST_PRODUCER = const.Declare tensor<1x64x250x250xf16, {order = #NHWC}> =
        dense<1.0> : tensor<1x64x250x250xf16>, [#const.Reorder<#NHWC>]

    %TILING = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 0, 125, 0]
        ]
    } : tensor<1x32x125x250xf16, {order = #NHWC}>,
        tensor<1x32x125x250xf16, {order = #NHWC}>
            -> tensor<1x32x250x250xf16, {order = #NHWC}>

    %MAIN_CONCAT = VPU.Concat(%CST_PRODUCER, %TILING) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 64, 0, 0]
        ]
    } : tensor<1x64x250x250xf16, {order = #NHWC}>,
        tensor<1x32x250x250xf16, {order = #NHWC}>
            -> tensor<1x96x250x250xf16, {order = #NHWC}>

    return %MAIN_CONCAT : tensor<1x96x250x250xf16, {order = #NHWC}>

    // CHECK:   [[CST_PRODUCER:%.*]] = const.Declare
    // CHECK:   [[TILING:%.*]] = VPU.Concat(%arg0, %arg1)
    // CHECK:   [[MAIN_CONCAT:%.*]] = VPU.Concat([[CST_PRODUCER]], [[TILING]])

    // CHECK:   return [[MAIN_CONCAT]] : tensor<1x96x250x250xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed0 = !VPU.DistributedTensor<
    1x32x128x128xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 32, 64, 128], [1, 32, 64, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    memory_shapes = [[1, 32, 65, 128], [1, 32, 65, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]
}>

!InputDistributed1 = !VPU.DistributedTensor<
    1x16x128x128xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    memory_shapes = [[1, 16, 65, 128], [1, 16, 65, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x48x128x128xf16, #NHWC, @CMX_NN, {
    mode = "OVERLAPPED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 48, 65, 128], [1, 48, 65, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]],
    memory_shapes = [[1, 48, 65, 128], [1, 48, 65, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]
}>

// CHECK-LABEL: ConcatWithExplicitOverlappedDistributedTensorType
func.func @ConcatWithExplicitOverlappedDistributedTensorType(%arg0: !InputDistributed0,
                                                             %arg1: !InputDistributed1)
    -> !OutputDistributed {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 32, 0, 0]
        ]
    } : !InputDistributed0, !InputDistributed1 -> !OutputDistributed

    return %concat : !OutputDistributed

    // CHECK:        [[CONCAT:%.*]] = VPU.Concat(%arg0, %arg1)
    // CHECK-SAME:         !VPU.DistributedTensor<1x32x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 64, 128], [1, 32, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 65, 128], [1, 32, 65, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]

    // CHECK-SAME:         !VPU.DistributedTensor<1x16x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 65, 128], [1, 16, 65, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]

    // CHECK-SAME:         -> !VPU.DistributedTensor<1x48x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "OVERLAPPED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 48, 65, 128], [1, 48, 65, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 48, 65, 128], [1, 48, 65, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 63, 0]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!InputDistributed0 = !VPU.DistributedTensor<
    1x32x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 32, 64, 128], [1, 32, 64, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    memory_shapes = [[1, 32, 64, 128], [1, 32, 64, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]
}>

!InputDistributed1 = !VPU.DistributedTensor<
    1x16x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    memory_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]
}>

!OutputDistributed = !VPU.DistributedTensor<
    1x48x128x128xf16, #NHWC, @CMX_NN, {
    mode = "SEGMENTED",
    num_tiles = [1, 1, 2, 1],
    num_clusters = 2 : i64,
    compute_shapes = [[1, 48, 64, 128], [1, 48, 64, 128]],
    compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    memory_shapes = [[1, 48, 64, 128], [1, 48, 64, 128]],
    memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]
}>

// CHECK-LABEL: ConcatWithExplicitSegmentedDistributedTensorType
func.func @ConcatWithExplicitSegmentedDistributedTensorType(%arg0: !InputDistributed0,
                                                             %arg1: !InputDistributed1)
    -> !OutputDistributed {

    %concat = VPU.Concat(%arg0, %arg1) {
        static_offsets = [
            [0, 0, 0, 0],
            [0, 32, 0, 0]
        ]
    } : !InputDistributed0, !InputDistributed1 -> !OutputDistributed

    return %concat : !OutputDistributed

    // CHECK:        [[CONCAT:%.*]] = VPU.Concat(%arg0, %arg1)
    // CHECK-SAME:         !VPU.DistributedTensor<1x32x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 32, 64, 128], [1, 32, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 32, 64, 128], [1, 32, 64, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]

    // CHECK-SAME:         !VPU.DistributedTensor<1x16x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 16, 64, 128], [1, 16, 64, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]

    // CHECK-SAME:         -> !VPU.DistributedTensor<1x48x128x128xf16, #NHWC, @CMX_NN
    // CHECK-SAME:             mode = "SEGMENTED"
    // CHECK-SAME:             num_tiles = [1, 1, 2, 1], num_clusters = 2 : i64
    // CHECK-SAME{LITERAL}:    compute_shapes = [[1, 48, 64, 128], [1, 48, 64, 128]], compute_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]],
    // CHECK-SAME{LITERAL}:    memory_shapes = [[1, 48, 64, 128], [1, 48, 64, 128]], memory_offsets = [[0, 0, 0, 0], [0, 0, 64, 0]]
}
