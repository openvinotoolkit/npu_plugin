//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --handle-large-kernels %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

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

    // CHECK:       [[POOL0:%.*]] = IE.AvgPool(%arg0)
    // CHECK-SAME:   {kernel_size = [1, 7], pads_begin = [0, 0], pads_end = [0, 1], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 7]}
    // CHECK:       [[POOL1:%.*]] = IE.AvgPool([[POOL0]])
    // CHECK-SAME:   {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK:       return [[POOL1]] : tensor<1x64x10x1xf16>

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

    // CHECK:       [[POOL0:%.*]] = IE.AvgPool(%arg0)
    // CHECK-SAME:   {kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [1, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [7, 1]}
    // CHECK:       [[POOL1:%.*]] = IE.AvgPool([[POOL0]])
    // CHECK-SAME:   {kernel_size = [2, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]}
    // CHECK:       return [[POOL1]] : tensor<1x64x1x10xf16>

}

// -----

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

    // CHECK:       [[AVGPOOL0:%.*]] = IE.AvgPool(%arg0)
    // CHECK-SAME:   {kernel_size = [4, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<CEIL>, strides = [4, 8]}
    // CHECK:       [[AVGPOOL1:%.*]] = IE.AvgPool([[AVGPOOL0]])
    // CHECK-SAME:   {kernel_size = [4, 4], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<CEIL>, strides = [4, 4]}
    // CHECK: return [[AVGPOOL1]] : tensor<1x1024x2x2xf16>
}

// -----

// CHECK-LABEL: @HandleLargerKernelsAvgPoolPaddingNeededOneDim
func.func @HandleLargerKernelsAvgPoolPaddingNeededOneDim(%arg0 : tensor<1x128x1x75076xf16>) -> (tensor<1x128x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 75076],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = #IE.rounding_type<FLOOR>,
        strides = [1, 1]
    } : tensor<1x128x1x75076xf16> -> tensor<1x128x1x1xf16>

    return %ave_pool : tensor<1x128x1x1xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 4]
    // CHECK-SAME:      : tensor<1x128x1x75076xf16> -> tensor<1x128x1x18769xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 10]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 10]
    // CHECK-SAME:      : tensor<1x128x1x18769xf16> -> tensor<1x128x1x1877xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 6]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 6]
    // CHECK-SAME:      : tensor<1x128x1x1877xf16> -> tensor<1x128x1x313xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 2]
    // CHECK-SAME:      : tensor<1x128x1x313xf16> -> tensor<1x128x1x157xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 2]
    // CHECK-SAME:      : tensor<1x128x1x157xf16> -> tensor<1x128x1x79xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 10]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 10]
    // CHECK-SAME:      : tensor<1x128x1x79xf16> -> tensor<1x128x1x8xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x128x1x8xf16> -> tensor<1x128x1x1xf16>
}
