//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --handle-large-kernels %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @HandleLargeKernelsAvgPool
func @HandleLargeKernelsAvgPool(%arg0 : tensor<1x2048x23x30xf16>) -> (tensor<1x2048x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [23, 30],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [23, 30]
    } : tensor<1x2048x23x30xf16> -> tensor<1x2048x1x1xf16>

    return %ave_pool : tensor<1x2048x1x1xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [6, 6]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [6, 6]
    // CHECK-SAME:      : tensor<1x2048x23x30xf16> -> tensor<1x2048x4x5xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [4, 5]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x2048x4x5xf16> -> tensor<1x2048x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsMaxPool
func @HandleLargeKernelsMaxPool(%arg0 : tensor<1x512x19x19xf16>) -> (tensor<1x512x19x19xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [13, 13],
        pads_begin = [6, 6],
        pads_end = [6, 6],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>

    return %max_pool : tensor<1x512x19x19xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [7, 7]
    // CHECK-SAME:      pads_begin = [3, 3]
    // CHECK-SAME:      pads_end = [3, 3]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [7, 7]
    // CHECK-SAME:      pads_begin = [3, 3]
    // CHECK-SAME:      pads_end = [3, 3]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x512x19x19xf16> -> tensor<1x512x19x19xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsXMaxPool
func @HandleLargeKernelsXMaxPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x1xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [1, 13],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [1, 13]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x1xf16>

    return %max_pool : tensor<1x64x10x1xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [1, 7]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [1, 7]
    // CHECK-SAME:      : tensor<1x64x10x13xf16> -> tensor<1x64x10x2xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [1, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [1, 2]
    // CHECK-SAME:      : tensor<1x64x10x2xf16> -> tensor<1x64x10x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsYMaxPool
func @HandleLargeKernelsYMaxPool(%arg0 : tensor<1x64x13x10xf16>) -> (tensor<1x64x1x10xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [13, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [13, 1]
    } : tensor<1x64x13x10xf16> -> tensor<1x64x1x10xf16>

    return %max_pool : tensor<1x64x1x10xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [7, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [7, 1]
    // CHECK-SAME:      : tensor<1x64x13x10xf16> -> tensor<1x64x2x10xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [2, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [2, 1]
    // CHECK-SAME:      : tensor<1x64x2x10xf16> -> tensor<1x64x1x10xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsGlobalMaxPool
func @HandleLargeKernelsGlobalMaxPool(%arg0 : tensor<1x128x9x16xf16>) -> (tensor<1x128x1x1xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        kernel_size = [9, 16],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [9, 16]
    } : tensor<1x128x9x16xf16> -> tensor<1x128x1x1xf16>

    return %max_pool : tensor<1x128x1x1xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [9, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [1, 4]
    // CHECK-SAME:      : tensor<1x128x9x16xf16> -> tensor<1x128x1x4xf16>
    // CHECK:       IE.MaxPool
    // CHECK-SAME:      kernel_size = [1, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x128x1x4xf16> -> tensor<1x128x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsGlobalAvgPool
func @HandleLargeKernelsGlobalAvgPool(%arg0 : tensor<1x128x9x16xf16>) -> (tensor<1x128x1x1xf16>) {
    %avg_pool = IE.AvgPool(%arg0) {
        kernel_size = [9, 16],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [9, 16]
    } : tensor<1x128x9x16xf16> -> tensor<1x128x1x1xf16>

    return %avg_pool : tensor<1x128x1x1xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [9, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [1, 4]
    // CHECK-SAME:      : tensor<1x128x9x16xf16> -> tensor<1x128x1x4xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x128x1x4xf16> -> tensor<1x128x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargerKernelsAvgPool
func @HandleLargerKernelsAvgPool(%arg0 : tensor<1x16x128x128xf16>) -> (tensor<1x16x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [128, 128],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [128, 128]
    } : tensor<1x16x128x128xf16> -> tensor<1x16x1x1xf16>

    return %ave_pool : tensor<1x16x1x1xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [8, 8]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [8, 8]
    // CHECK-SAME:      : tensor<1x16x128x128xf16> -> tensor<1x16x16x16xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [4, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [4, 4]
    // CHECK-SAME:      : tensor<1x16x16x16xf16> -> tensor<1x16x4x4xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [4, 4]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x16x4x4xf16> -> tensor<1x16x1x1xf16>
}

// -----

// CHECK-LABEL: @HandleSymmetricLargeKernelsGlobalAvgPool
func @HandleSymmetricLargeKernelsGlobalAvgPool(%arg0 : tensor<1x2048x65x65xf16>) -> (tensor<1x2048x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [65, 65],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [65, 65]
    } : tensor<1x2048x65x65xf16> -> tensor<1x2048x1x1xf16>

    return %ave_pool : tensor<1x2048x1x1xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [6, 6]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [6, 6]
    // CHECK-SAME:      : tensor<1x2048x65x65xf16> -> tensor<1x2048x11x11xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [11, 11]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x2048x11x11xf16> -> tensor<1x2048x1x1xf16>
}
