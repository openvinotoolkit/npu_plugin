//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --convert-shape-to-4d --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @ConvertShapeTo4DFrom5D
func @ConvertShapeTo4DFrom5D(%arg0: tensor<1x3x9x16x1xf16>, %arg1: tensor<1x1x1x1x1xf16>) -> (tensor<1x3x9x16x1xf16>) {
    %0 = IE.Sigmoid(%arg0) : tensor<1x3x9x16x1xf16> -> tensor<1x3x9x16x1xf16>
    %1 = IE.Subtract(%0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x3x9x16x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x3x9x16x1xf16>
    return %1 : tensor<1x3x9x16x1xf16>
    // CHECK:    %[[Reshape_0:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [3, 9, 16, 1]} : tensor<1x3x9x16x1xf16> -> tensor<3x9x16x1xf16>
    // CHECK:    %[[SIGMOID:.*]] = IE.Sigmoid(%[[Reshape_0]]) : tensor<3x9x16x1xf16> -> tensor<3x9x16x1xf16>
    // CHECK:    %[[Reshape_1:.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 1, 1, 1]} : tensor<1x1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:    %[[SUB:.*]] = IE.Subtract(%[[SIGMOID]], %[[Reshape_1]]) {auto_broadcast = "NUMPY"} : tensor<3x9x16x1xf16>, tensor<1x1x1x1xf16> -> tensor<3x9x16x1xf16>
    // CHECK:    %[[Reshape_2:.*]] = IE.AffineReshape(%[[SUB]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 3, 9, 16, 1]} : tensor<3x9x16x1xf16> -> tensor<1x3x9x16x1xf16>
    // CHECK:    return %[[Reshape_2]]
}

// CHECK-LABEL: func @ConvertShapeTo4DFromStridedSlice
func @ConvertShapeTo4DFromStridedSlice(%arg0: tensor<4004x320xf16>) -> (tensor<4004x160xf16>) {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 1], begins_attr = [0, 0], ellipsis_mask = [], end_mask = [1, 0], ends_attr = [4004, 320], new_axis_mask = [], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [], strides_attr = [1, 2]} : tensor<4004x320xf16> -> tensor<4004x160xf16>
    return %0 : tensor<4004x160xf16>
    // CHECK:    %[[Reshape_0:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1, 2, 3]], shape_value = [4004, 320, 1, 1]} : tensor<4004x320xf16> -> tensor<4004x320x1x1xf16>
    // CHECK:    %[[STRIDEDSLICE:.*]] = IE.StridedSlice(%0) {begin_mask = [0, 1], begins_attr = [0, 0], ellipsis_mask = [], end_mask = [1, 0], ends_attr = [4004, 320], new_axis_mask = [], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [], strides_attr = [1, 2]} : tensor<4004x320x1x1xf16> -> tensor<4004x160x1x1xf16>
    // CHECK:    %[[Reshape_1:.*]] = IE.AffineReshape(%[[STRIDEDSLICE]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1], [1], [1]], shape_value = [4004, 160]} : tensor<4004x160x1x1xf16> -> tensor<4004x160xf16>
    // CHECK:    return %[[Reshape_1]]
}
