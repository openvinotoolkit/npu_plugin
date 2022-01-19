// RUN: vpux-opt --split-input-file --handle-large-kernels %s | FileCheck %s

// CHECK-LABEL: @HandleLargeKernelsAvgPool
func @HandleLargeKernelsAvgPool(%arg0 : tensor<1x2048x23x30xf16>) -> (tensor<1x2048x1x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        exclude_pads,
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
        exclude_pads,
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

// CHECK-LABEL: @HandleLargeKernelsXAvgPool
func @HandleLargeKernelsXAvgPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [1, 13],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [1, 13]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x1xf16>

    return %ave_pool : tensor<1x64x10x1xf16>
    // CHECK:       IE.Slice
    // CHECK-SAME:    tensor<1x64x10x13xf16> to tensor<1x64x10x13xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 7]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [7, 7]
    // CHECK-SAME:      : tensor<1x64x10x13xf16> -> tensor<1x64x2x2xf16>
    // CHECK:       IE.Slice
    // CHECK-SAME:    tensor<1x64x10x13xf16> to tensor<1x64x9x13xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 7]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 1]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [7, 7]
    // CHECK-SAME:      : tensor<1x64x9x13xf16> -> tensor<1x64x2x2xf16>
    // CHECK:       IE.Concat
    // CHECK-SAME:      : tensor<1x64x2x2xf16>, tensor<1x64x2x2xf16>, tensor<1x64x2x2xf16>, tensor<1x64x1x2xf16>, tensor<1x64x1x2xf16>, tensor<1x64x1x2xf16>, tensor<1x64x1x2xf16> -> tensor<1x64x10x2xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [1, 2]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x64x10x2xf16> -> tensor<1x64x10x1xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsXMaxPool
func @HandleLargeKernelsXMaxPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x1xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        exclude_pads,
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

// CHECK-LABEL: @HandleLargeKernelsYAvgPool
func @HandleLargeKernelsYAvgPool(%arg0 : tensor<1x64x13x10xf16>) -> (tensor<1x64x1x10xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [13, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [13, 1]
    } : tensor<1x64x13x10xf16> -> tensor<1x64x1x10xf16>
    return %ave_pool : tensor<1x64x1x10xf16>
    // CHECK:       IE.Slice
    // CHECK-SAME:    tensor<1x64x13x10xf16> to tensor<1x64x13x10xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [7, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [7, 7]
    // CHECK-SAME:      : tensor<1x64x13x10xf16> -> tensor<1x64x2x2xf16>
    // CHECK:       IE.Slice
    // CHECK-SAME:    tensor<1x64x13x10xf16> to tensor<1x64x13x9xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [7, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [1, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [7, 7]
    // CHECK-SAME:      : tensor<1x64x13x9xf16> -> tensor<1x64x2x2xf16>
    // CHECK:       IE.Concat
    // CHECK-SAME:      : tensor<1x64x2x2xf16>, tensor<1x64x2x2xf16>, tensor<1x64x2x2xf16>, tensor<1x64x2x1xf16>, tensor<1x64x2x1xf16>, tensor<1x64x2x1xf16>, tensor<1x64x2x1xf16> -> tensor<1x64x2x10xf16>
    // CHECK:       IE.AvgPool
    // CHECK-SAME:      kernel_size = [2, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = "FLOOR",
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      : tensor<1x64x2x10xf16> -> tensor<1x64x1x10xf16>
}

// -----

// CHECK-LABEL: @HandleLargeKernelsYMaxPool
func @HandleLargeKernelsYMaxPool(%arg0 : tensor<1x64x13x10xf16>) -> (tensor<1x64x1x10xf16>) {
    %max_pool = IE.MaxPool(%arg0) {
        exclude_pads,
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
