// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --handle-large-kernels %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

// CHECK-LABEL: @HandleLargeKernelsXAvgPool
func @HandleLargeKernelsXAvgPool(%arg0 : tensor<1x64x10x13xf16>) -> (tensor<1x64x10x1xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [1, 13],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [1, 13]
    } : tensor<1x64x10x13xf16> -> tensor<1x64x10x1xf16>

    return %ave_pool : tensor<1x64x10x1xf16>

    // CHECK:       [[POOL0:%.*]] = IE.AvgPool(%arg0)
    // CHECK-SAME:   {kernel_size = [1, 7], pads_begin = [0, 0], pads_end = [0, 1], rounding_type = "FLOOR", strides = [1, 7]}
    // CHECK:       [[POOL1:%.*]] = IE.AvgPool([[POOL0]])
    // CHECK-SAME:   {kernel_size = [1, 2], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]}
    // CHECK:       return [[POOL1]] : tensor<1x64x10x1xf16>

}

// -----

// CHECK-LABEL: @HandleLargeKernelsYAvgPool
func @HandleLargeKernelsYAvgPool(%arg0 : tensor<1x64x13x10xf16>) -> (tensor<1x64x1x10xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [13, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [13, 1]
    } : tensor<1x64x13x10xf16> -> tensor<1x64x1x10xf16>

    return %ave_pool : tensor<1x64x1x10xf16>

    // CHECK:       [[POOL0:%.*]] = IE.AvgPool(%arg0)
    // CHECK-SAME:   {kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [1, 0], rounding_type = "FLOOR", strides = [7, 1]}
    // CHECK:       [[POOL1:%.*]] = IE.AvgPool([[POOL0]])
    // CHECK-SAME:   {kernel_size = [2, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]}
    // CHECK:       return [[POOL1]] : tensor<1x64x1x10xf16>

}

// -----

// -----

// CHECK-LABEL: @HandleLargeKernelsSplitAvgPool
func @HandleLargeKernelsSplitAvgPool(%arg0 : tensor<1x1024x32x64xf16>) -> (tensor<1x1024x2x2xf16>) {
    %ave_pool = IE.AvgPool(%arg0) {
        kernel_size = [16, 32],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "CEIL",
        strides = [16, 32]
    } : tensor<1x1024x32x64xf16> -> tensor<1x1024x2x2xf16>

    return %ave_pool : tensor<1x1024x2x2xf16>

    // CHECK:       [[AVGPOOL0:%.*]] = IE.AvgPool(%arg0)
    // CHECK-SAME:   {kernel_size = [4, 8], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "CEIL", strides = [4, 8]}
    // CHECK:       [[AVGPOOL1:%.*]] = IE.AvgPool([[AVGPOOL0]])
    // CHECK-SAME:   {kernel_size = [4, 4], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "CEIL", strides = [4, 4]}
    // CHECK: return [[AVGPOOL1]] : tensor<1x1024x2x2xf16>
}
