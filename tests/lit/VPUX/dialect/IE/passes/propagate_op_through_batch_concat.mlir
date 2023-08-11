//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-op-through-batch-concat %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

// CHECK-LABEL: @PropagateSoftmaxThroughBatchUnrolledMatmul
func.func @PropagateSoftmaxThroughBatchUnrolledMatmul(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.MatMul(%arg1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %3 = IE.Concat(%1, %2) {per_axis = {axis = 0 : i64}} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<32x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [2, 16, 2]} : tensor<32x2xf32> -> tensor<2x16x2xf32>
    %5 = IE.SoftMax(%4) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    return %5 : tensor<2x16x2xf32>

    // CHECK:       %[[MATMUL_1:.*]] = IE.MatMul
    // CHECK:       %[[MATMUL_2:.*]] = IE.MatMul
    // CHECK:       %[[SOFTMAX_1:.*]] = IE.SoftMax(%[[MATMUL_1]]) {axisInd = 1 : i64} : tensor<16x2xf32> -> tensor<16x2xf32>
    // CHECK:       %[[SOFTMAX_2:.*]] = IE.SoftMax(%[[MATMUL_2]]) {axisInd = 1 : i64} : tensor<16x2xf32> -> tensor<16x2xf32>
    // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[SOFTMAX_1]], %[[SOFTMAX_2]]) {per_axis = {axis = 0 : i64}} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<32x2xf32>
    // CHECK:       %[[OUT:.*]] = IE.Reshape(%[[CONCAT]]) {shape_value = [2, 16, 2]} : tensor<32x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       return %[[OUT]] : tensor<2x16x2xf32>
}

// CHECK-LABEL: @PropagateSoftmaxonAxis0
func.func @PropagateSoftmaxonAxis0(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<16x2x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.MatMul(%arg1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %3 = IE.Concat(%1, %2) {per_axis = {axis = 1 : i64}} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<16x4xf32>
    %4 = IE.Reshape(%3) {shape_value = [16, 2, 2]} : tensor<16x4xf32> -> tensor<16x2x2xf32>
    %5 = IE.SoftMax(%4) {axisInd = 0 : i64} : tensor<16x2x2xf32> -> tensor<16x2x2xf32>
    return %5 : tensor<16x2x2xf32>

    // CHECK:       %[[MATMUL_1:.*]] = IE.MatMul
    // CHECK:       %[[MATMUL_2:.*]] = IE.MatMul
    // CHECK:       %[[SOFTMAX_1:.*]] = IE.SoftMax(%[[MATMUL_1]]) {axisInd = 0 : i64} : tensor<16x2xf32> -> tensor<16x2xf32>
    // CHECK:       %[[SOFTMAX_2:.*]] = IE.SoftMax(%[[MATMUL_2]]) {axisInd = 0 : i64} : tensor<16x2xf32> -> tensor<16x2xf32>
    // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[SOFTMAX_1]], %[[SOFTMAX_2]]) {per_axis = {axis = 1 : i64}} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<16x4xf32>
    // CHECK:       %[[OUT:.*]] = IE.Reshape(%[[CONCAT]]) {shape_value = [16, 2, 2]} : tensor<16x4xf32> -> tensor<16x2x2xf32>
    // CHECK:       return %[[OUT]] : tensor<16x2x2xf32>
}

// CHECK-LABEL: @NoPropagateSoftmaxForConcatHasNonMatmulInput
func.func @NoPropagateSoftmaxForConcatHasNonMatmulInput(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %cst_1 = const.Declare tensor<16x2xf32> = dense<1.000000e+00> : tensor<16x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.Add(%arg1, %cst_1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<16x2xf32>
    %3 = IE.Concat(%1, %2) {per_axis = {axis = 0 : i64}} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<32x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [2, 16, 2]} : tensor<32x2xf32> -> tensor<2x16x2xf32>
    %5 = IE.SoftMax(%4) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    return %5 : tensor<2x16x2xf32>

    // CHECK:       %[[MATMUL:.*]] = IE.MatMul
    // CHECK:       %[[ADD:.*]] = IE.Add
    // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[MATMUL]], %[[ADD]]) {per_axis = {axis = 0 : i64}} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<32x2xf32>
    // CHECK:       %[[RESHAPE:.*]] = IE.Reshape(%[[CONCAT]]) {shape_value = [2, 16, 2]} : tensor<32x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       %[[SOFTMAX:.*]] = IE.SoftMax(%[[RESHAPE]]) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       return %[[SOFTMAX]] : tensor<2x16x2xf32>
}

// CHECK-LABEL: @NoPropagateForConcatWithoutPerAxisAttr
func.func @NoPropagateForConcatWithoutPerAxisAttr(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.MatMul(%arg1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0], [16, 0]]} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<32x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [2, 16, 2]} : tensor<32x2xf32> -> tensor<2x16x2xf32>
    %5 = IE.SoftMax(%4) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    return %5 : tensor<2x16x2xf32>

    // CHECK:       %[[MATMUL_1:.*]] = IE.MatMul
    // CHECK:       %[[MATMUL_2:.*]] = IE.MatMul
    // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[MATMUL_1]], %[[MATMUL_2]]) {static_offsets = {{\[\[}}0, 0], [16, 0]]} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<32x2xf32>
    // CHECK:       %[[RESHAPE:.*]] = IE.Reshape(%[[CONCAT]]) {shape_value = [2, 16, 2]} : tensor<32x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       %[[SOFTMAX:.*]] = IE.SoftMax(%[[RESHAPE]]) {axisInd = -1 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       return %[[SOFTMAX]] : tensor<2x16x2xf32>
}

// CHECK-LABEL: @NoPropagateWhenAxisConflict
func.func @NoPropagateWhenAxisConflict(%arg0: tensor<16x2xf32>, %arg1: tensor<16x2xf32>) -> tensor<2x16x2xf32> {
    %cst = const.Declare tensor<2x2xf32> = dense<1.000000e+00> : tensor<2x2xf32>
    %1 = IE.MatMul(%arg0, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %2 = IE.MatMul(%arg1, %cst) : tensor<16x2xf32>, tensor<2x2xf32> -> tensor<16x2xf32>
    %3 = IE.Concat(%1, %2) {per_axis = {axis = 0 : i64}} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<32x2xf32>
    %4 = IE.Reshape(%3) {shape_value = [2, 16, 2]} : tensor<32x2xf32> -> tensor<2x16x2xf32>
    %5 = IE.SoftMax(%4) {axisInd = 0 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    return %5 : tensor<2x16x2xf32>

    // CHECK:       %[[MATMUL_1:.*]] = IE.MatMul
    // CHECK:       %[[MATMUL_2:.*]] = IE.MatMul
    // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[MATMUL_1]], %[[MATMUL_2]]) {per_axis = {axis = 0 : i64}} : tensor<16x2xf32>, tensor<16x2xf32> -> tensor<32x2xf32>
    // CHECK:       %[[RESHAPE:.*]] = IE.Reshape(%[[CONCAT]]) {shape_value = [2, 16, 2]} : tensor<32x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       %[[SOFTMAX:.*]] = IE.SoftMax(%[[RESHAPE]]) {axisInd = 0 : i64} : tensor<2x16x2xf32> -> tensor<2x16x2xf32>
    // CHECK:       return %[[SOFTMAX]] : tensor<2x16x2xf32>
}
