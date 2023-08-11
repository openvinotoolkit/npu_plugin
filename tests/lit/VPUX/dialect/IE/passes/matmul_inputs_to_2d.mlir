//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --matmul-inputs-to-2d %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @MatMulInputsTo2d
func.func @MatMulInputsTo2d(%arg0: tensor<2x1x512xf32>) -> tensor<2x1x40xf32> {
    %cst = const.Declare tensor<2x512x40xf32> = dense<1.0> : tensor<2x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<2x1x512xf32>, tensor<2x512x40xf32> -> tensor<2x1x40xf32>

    return %0 : tensor<2x1x40xf32>

    // CHECK-DAG:       %[[CST:.*]] = const.Declare tensor<2x512x40xf32> = dense<1.000000e+00> : tensor<2x512x40xf32>
    // CHECK:       %[[IN_1:.*]] = IE.Slice %arg0 [0, 0, 0] [1, 1, 512] : tensor<2x1x512xf32> to tensor<1x1x512xf32>
    // CHECK:       %[[IN_1_2D:.*]] = IE.Reshape(%[[IN_1]]) {shape_value = [1, 512]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       %[[IN_2:.*]] = IE.Slice %arg0 [1, 0, 0] [1, 1, 512] : tensor<2x1x512xf32> to tensor<1x1x512xf32>
    // CHECK:       %[[IN_2_2D:.*]] = IE.Reshape(%[[IN_2]]) {shape_value = [1, 512]} : tensor<1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       %[[CST_1:.*]] = IE.Slice %[[CST]] [0, 0, 0] [1, 512, 40] : tensor<2x512x40xf32> to tensor<1x512x40xf32>
    // CHECK:       %[[CST_1_2D:.*]] = IE.Reshape(%[[CST_1]]) {shape_value = [512, 40]} : tensor<1x512x40xf32> -> tensor<512x40xf32>
    // CHECK:       %[[CST_2:.*]] = IE.Slice %[[CST]] [1, 0, 0] [1, 512, 40] : tensor<2x512x40xf32> to tensor<1x512x40xf32>
    // CHECK:       %[[CST_2_2D:.*]] = IE.Reshape(%[[CST_2]]) {shape_value = [512, 40]} : tensor<1x512x40xf32> -> tensor<512x40xf32>
    // CHECK:       %[[MATMUL_1:.*]] = IE.MatMul(%[[IN_1_2D]], %[[CST_1_2D]]) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    // CHECK:       %[[MATMUL_2:.*]] = IE.MatMul(%[[IN_2_2D]], %[[CST_2_2D]]) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[MATMUL_1]], %[[MATMUL_2]]) {per_axis = {axis = 0 : i64}} : tensor<1x40xf32>, tensor<1x40xf32> -> tensor<2x40xf32>
    // CHECK:       %[[OUT:.*]] = IE.Reshape(%[[CONCAT]]) {shape_value = [2, 1, 40]} : tensor<2x40xf32> -> tensor<2x1x40xf32>
    // CHECK:       return %[[OUT]] : tensor<2x1x40xf32>
}

// CHECK-LABEL: @MatMul4dInputsTo2d
func.func @MatMul4dInputsTo2d(%arg0: tensor<1x2x1x512xf32>) -> tensor<1x2x1x40xf32> {
    %cst = const.Declare tensor<1x2x512x40xf32> = dense<1.0> : tensor<1x2x512x40xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x2x1x512xf32>, tensor<1x2x512x40xf32> -> tensor<1x2x1x40xf32>

    return %0 : tensor<1x2x1x40xf32>

    // CHECK-DAG:       %[[CST:.*]] = const.Declare tensor<1x2x512x40xf32> = dense<1.000000e+00> : tensor<1x2x512x40xf32>
    // CHECK:       %[[IN_1:.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:       %[[IN_1_2D:.*]] = IE.Reshape(%[[IN_1]]) {shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       %[[IN_2:.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 1, 512] : tensor<1x2x1x512xf32> to tensor<1x1x1x512xf32>
    // CHECK:       %[[IN_2_2D:.*]] = IE.Reshape(%[[IN_2]]) {shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>
    // CHECK:       %[[CST_1:.*]] = IE.Slice %[[CST]] [0, 0, 0, 0] [1, 1, 512, 40] : tensor<1x2x512x40xf32> to tensor<1x1x512x40xf32>
    // CHECK:       %[[CST_1_2D:.*]] = IE.Reshape(%[[CST_1]]) {shape_value = [512, 40]} : tensor<1x1x512x40xf32> -> tensor<512x40xf32>
    // CHECK:       %[[CST_2:.*]] = IE.Slice %[[CST]] [0, 1, 0, 0] [1, 1, 512, 40] : tensor<1x2x512x40xf32> to tensor<1x1x512x40xf32>
    // CHECK:       %[[CST_2_2D:.*]] = IE.Reshape(%[[CST_2]]) {shape_value = [512, 40]} : tensor<1x1x512x40xf32> -> tensor<512x40xf32>
    // CHECK:       %[[MATMUL_1:.*]] = IE.MatMul(%[[IN_1_2D]], %[[CST_1_2D]]) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    // CHECK:       %[[MATMUL_2:.*]] = IE.MatMul(%[[IN_2_2D]], %[[CST_2_2D]]) : tensor<1x512xf32>, tensor<512x40xf32> -> tensor<1x40xf32>
    // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[MATMUL_1]], %[[MATMUL_2]]) {per_axis = {axis = 0 : i64}} : tensor<1x40xf32>, tensor<1x40xf32> -> tensor<2x40xf32>
    // CHECK:       %[[OUT:.*]] = IE.Reshape(%[[CONCAT]]) {shape_value = [1, 2, 1, 40]} : tensor<2x40xf32> -> tensor<1x2x1x40xf32>
    // CHECK:       return %[[OUT]] : tensor<1x2x1x40xf32>
}

// CHECK-LABEL: @MatMul3dInputsBatch1To2d
func.func @MatMul3dInputsBatch1To2d(%arg0: tensor<1x1x1024xf32>) -> tensor<1x1x512xf32> {
    %cst = const.Declare tensor<1x1024x512xf32> = dense<1.0> : tensor<1x1024x512xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x1x1024xf32>, tensor<1x1024x512xf32> -> tensor<1x1x512xf32>

    return %0 : tensor<1x1x512xf32>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<1x1024x512xf32> = dense<1.000000e+00> : tensor<1x1024x512xf32>
    // CHECK: %[[RESHAPE0:.*]] = IE.Reshape(%arg0) {shape_value = [1, 1024]} : tensor<1x1x1024xf32> -> tensor<1x1024xf32>
    // CHECK: %[[RESHAPE1:.*]] = IE.Reshape(%[[CST]]) {shape_value = [1024, 512]} : tensor<1x1024x512xf32> -> tensor<1024x512xf32>
    // CHECK: %[[MTML:.*]] = IE.MatMul(%[[RESHAPE0]], %[[RESHAPE1]]) : tensor<1x1024xf32>, tensor<1024x512xf32> -> tensor<1x512xf32>
    // CHECK: %[[OUT:.*]] = IE.Reshape(%[[MTML]]) {shape_value = [1, 1, 512]} : tensor<1x512xf32> -> tensor<1x1x512xf32>
    // CHECK: return %[[OUT]] : tensor<1x1x512xf32>
}

// CHECK-LABEL: @NoChangesMatMul3dInput2DWeightsTo2d
func.func @NoChangesMatMul3dInput2DWeightsTo2d(%arg0: tensor<1x4x9728xf32>) -> tensor<1x4x512xf32> {
    %cst = const.Declare tensor<9728x512xf32> = dense<1.0> : tensor<9728x512xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x4x9728xf32>, tensor<9728x512xf32> -> tensor<1x4x512xf32>

    return %0 : tensor<1x4x512xf32>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<9728x512xf32> = dense<1.000000e+00> : tensor<9728x512xf32>
    // CHECK: %[[RESHAPE:.*]] = IE.Reshape(%arg0) {shape_value = [4, 9728]} : tensor<1x4x9728xf32> -> tensor<4x9728xf32>
    // CHECK: %[[MATMUL:.*]] = IE.MatMul(%[[RESHAPE]], %[[CST]]) : tensor<4x9728xf32>, tensor<9728x512xf32> -> tensor<4x512xf32>
    // CHECK: %[[OUT:.*]] = IE.Reshape(%[[MATMUL]]) {shape_value = [1, 4, 512]} : tensor<4x512xf32> -> tensor<1x4x512xf32>
    // CHECK: return %[[OUT]] : tensor<1x4x512xf32>
}

// CHECK-LABEL: @NoChangesMatMul3dInputWithMulChannel2DWeightsTo2d
func.func @NoChangesMatMul3dInputWithMulChannel2DWeightsTo2d(%arg0: tensor<2x64x128xf32>) -> tensor<2x64x64xf32> {
    %cst = const.Declare tensor<128x64xf32> = dense<1.0> : tensor<128x64xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<2x64x128xf32>, tensor<128x64xf32> -> tensor<2x64x64xf32>

    return %0 : tensor<2x64x64xf32>

    // CHECK-DAG: %[[CST:.*]] = const.Declare tensor<128x64xf32> = dense<1.000000e+00> : tensor<128x64xf32>
    // CHECK: %[[OUT:.*]] = IE.MatMul(%arg0, %[[CST]]) : tensor<2x64x128xf32>, tensor<128x64xf32> -> tensor<2x64x64xf32>
    // CHECK: return %[[OUT]] : tensor<2x64x64xf32>
}


// CHECK-LABEL: @MatMul4dInputWithMulChannel3dWeightsTo2d
func.func @MatMul4dInputWithMulChannel3dWeightsTo2d(%arg0: tensor<1x2x16x2xf32>) -> tensor<1x2x16x2xf32> {
    %cst = const.Declare tensor<1x2x2xf32> = dense<1.0> : tensor<1x2x2xf32>
    %0 = IE.MatMul(%arg0, %cst) : tensor<1x2x16x2xf32>, tensor<1x2x2xf32> -> tensor<1x2x16x2xf32>

    return %0 : tensor<1x2x16x2xf32>

    // CHECK-DAG:       %[[CST:.*]] = const.Declare tensor<1x2x2xf32> = dense<1.000000e+00> : tensor<1x2x2xf32>
    // CHECK:       %[[IN_1:.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 16, 2] : tensor<1x2x16x2xf32> to tensor<1x1x16x2xf32>
    // CHECK:       %[[IN_1_2D:.*]] = IE.Reshape(%[[IN_1]]) {shape_value = [16, 2]} : tensor<1x1x16x2xf32> -> tensor<16x2xf32>
    // CHECK:       %[[IN_2:.*]] = IE.Slice %arg0 [0, 1, 0, 0] [1, 1, 16, 2] : tensor<1x2x16x2xf32> to tensor<1x1x16x2xf32>
    // CHECK:       %[[IN_2_2D:.*]] = IE.Reshape(%[[IN_2]]) {shape_value = [16, 2]} : tensor<1x1x16x2xf32> -> tensor<16x2xf32>
    // CHECK:       %[[CST_2D:.*]] = IE.Reshape(%[[CST]]) {shape_value = [2, 2]} : tensor<1x2x2xf32> -> tensor<2x2xf32>
    // CHECK:       %[[MATMUL_1:.*]] = IE.MatMul(%[[IN_1_2D]], %[[CST_2D]]) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    // CHECK:       %[[MATMUL_2:.*]] = IE.MatMul(%[[IN_2_2D]], %[[CST_2D]]) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[MATMUL_1]], %[[MATMUL_2]]) {per_axis = {axis = 0 : i64}} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<32x2xf32>
    // CHECK:       %[[OUT:.*]] = IE.Reshape(%[[CONCAT]]) {shape_value = [1, 2, 16, 2]} : tensor<32x2xf32> -> tensor<1x2x16x2xf32>
    // CHECK:       return %[[OUT]] : tensor<1x2x16x2xf32>
}
