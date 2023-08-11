//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --convert-shape-to-4d --canonicalize %s | FileCheck %s

// CHECK-LABEL: func.func @ConvertShapeTo4DFrom5D
func.func @ConvertShapeTo4DFrom5D(%arg0: tensor<1x3x9x16x1xf16>, %arg1: tensor<1x1x1x1x1xf16>) -> (tensor<1x3x9x16x1xf16>) {
    %0 = IE.Sigmoid(%arg0) : tensor<1x3x9x16x1xf16> -> tensor<1x3x9x16x1xf16>
    %1 = IE.Subtract(%0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x9x16x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x3x9x16x1xf16>
    return %1 : tensor<1x3x9x16x1xf16>
    // CHECK:    %[[Reshape_0:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 3, 9, 16]} : tensor<1x3x9x16x1xf16> -> tensor<1x3x9x16xf16>
    // CHECK:    %[[Reshape_1:.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3], [3]], shape_value = [1, 1, 1, 1]} : tensor<1x1x1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:    %[[SIGMOID:.*]] = IE.Sigmoid(%[[Reshape_0]]) : tensor<1x3x9x16xf16> -> tensor<1x3x9x16xf16>
    // CHECK:    %[[SUB:.*]] = IE.Subtract(%[[SIGMOID]], %[[Reshape_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x3x9x16xf16>, tensor<1x1x1x1xf16> -> tensor<1x3x9x16xf16>
    // CHECK:    %[[Reshape_2:.*]] = IE.AffineReshape(%[[SUB]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2], [3, 4]], shape_value = [1, 3, 9, 16, 1]} : tensor<1x3x9x16xf16> -> tensor<1x3x9x16x1xf16>
    // CHECK:    return %[[Reshape_2]]
}

// -----

// CHECK-LABEL: func.func @ConvertGatherTo4D
func.func @ConvertGatherTo4D(%arg0: tensor<1x468x2xf16>) -> tensor<1x71x2xf16> {
    %cst = const.Declare tensor<71xsi32> = dense<1> : tensor<71xsi32>
    %0 = IE.Gather(%arg0, %cst) {axis_value = 1 : i64, batch_dims = 0 : i64} : tensor<1x468x2xf16>, tensor<71xsi32> -> tensor<1x71x2xf16>

    return %0 : tensor<1x71x2xf16>

    // CHECK-DAG:   [[CST:%.*]] = const.Declare tensor<71xsi32> = dense<1> : tensor<71xsi32>
    // CHECK:       [[RESHAPE_IN:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SMAE:         shape_value = [1, 1, 468, 2]
    // CHECK-SAME:     } : tensor<1x468x2xf16> -> tensor<1x1x468x2xf16>
    // CHECK:       [[GATHER:%.*]] = IE.Gather([[RESHAPE_IN]], [[CST]]) {axis_value = 2 : i64, batch_dims = 0 : i64} : tensor<1x1x468x2xf16>, tensor<71xsi32> -> tensor<1x1x71x2xf16>
    // CHECK:       [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[GATHER]]) {
    // CHECK-SMAE:         shape_value = [1, 71, 2]
    // CHECK-SAME:     } : tensor<1x1x71x2xf16> -> tensor<1x71x2xf16>
    // CHECK:    return [[RESHAPE_OUT]]
}
