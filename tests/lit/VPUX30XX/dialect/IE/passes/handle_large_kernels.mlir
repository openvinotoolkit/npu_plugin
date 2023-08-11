//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --handle-large-kernels %s | FileCheck %s

// CHECK-LABEL: @HandleLargeKernelsXAvgPool
func.func @HandleLargeKernelsXAvgPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 13],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 13]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x1xf16>

    return %ave_pool : tensor<1x64x10x1xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 13]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 13]
    // CHECK-SAME:      : tensor<1x64x10x13xf16> -> tensor<1x64x10x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsYAvgPool
func.func @HandleLargeKernelsYAvgPool(%arg0 : tensor<1x64x13x10xf16>) -> (tensor<1x64x1x10xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [13, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [13, 1]
    } : tensor<1x64x13x10xf16> -> tensor<1x64x1x10xf16>
    return %ave_pool : tensor<1x64x1x10xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [13, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [13, 1]
    // CHECK-SAME:      : tensor<1x64x13x10xf16> -> tensor<1x64x1x10xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsSplitAvgPool
func.func @HandleLargeKernelsSplitAvgPool(%arg0 : tensor<1x1024x32x64xf16>) -> (tensor<1x1024x2x2xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [16, 32],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<CEIL>,
        strides = [16, 32]
    } : tensor<1x1024x32x64xf16> -> tensor<1x1024x2x2xf16>
    return %ave_pool : tensor<1x1024x2x2xf16>
    // CHECK: [[AVGPOOL0:%.*]] = IE.AvgPool(%arg0) {kernel_size = [4, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 8]} :
    // CHECK-SAME: tensor<1x1024x32x64xf16> -> tensor<1x1024x4x8xf16>
    // CHECK: [[SLICE1_ARG1:%.*]] = IE.Slice %arg0 [0, 0, 4, 0] [1, 1024, 28, 64] : tensor<1x1024x32x64xf16> to tensor<1x1024x28x64xf16>
    // CHECK: [[AVGPOOL1:%.*]] = IE.AvgPool([[SLICE1_ARG1]]) {kernel_size = [4, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [8, 8]} :
    // CHECK-SAME: tensor<1x1024x28x64xf16> -> tensor<1x1024x4x8xf16>
    // CHECK: [[CONCAT0:%.*]] = IE.Concat([[AVGPOOL0]], [[AVGPOOL1]]) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} :
    // CHECK-SAME: tensor<1x1024x4x8xf16>, tensor<1x1024x4x8xf16> -> tensor<1x1024x8x8xf16>
    // CHECK: [[AVGPOOL2:%.*]] = IE.AvgPool([[CONCAT0]]) {kernel_size = [4, 4], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<CEIL>, strides = [4, 4]} :
    // CHECK-SAME: tensor<1x1024x8x8xf16> -> tensor<1x1024x2x2xf16>
    // CHECK: return [[AVGPOOL2]] : tensor<1x1024x2x2xf16>
}
