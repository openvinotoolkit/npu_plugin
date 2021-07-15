// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=VPU3700 compilation-mode=ReferenceHW" --fuse-post-ops %s | FileCheck %s

func @Conv2dWithReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x16x2x2xf16> = #const.Content<dense<1.0> : tensor<16x16x2x2xf16>>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1 : i32, 1 : i32],
            pads_begin = [0 : i32, 0 : i32],
            pads_end = [0 : i32, 0 : i32],
            dilations = [1 : i32, 1 : i32]
        } :
        tensor<1x16x4x4xf16>, tensor<16x16x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       %1 = IE.Convolution(%arg0, %0)
    // CHECK-SAME:     dilations = [1 : i32, 1 : i32],
    // CHECK-SAME:     pads_begin = [0 : i32, 0 : i32],
    // CHECK-SAME:     pads_end = [0 : i32, 0 : i32],
    // CHECK-SAME:     post_op = {kind = "RELU", params = {}},
    // CHECK-SAME:     strides = [1 : i32, 1 : i32]
    // CHECK-NOT:   IE.ReLU
}

// -----

func @MaxPoolWithReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %0 = IE.MaxPool(%arg0)
         {
             kernel_size = [2 : i32, 2 : i32],
             pads_begin = [0 : i32, 0 : i32],
             pads_end = [0 : i32, 0 : i32],
             strides = [1 : i32, 1 : i32],
             rounding_type = "CEIL"
         } :
         tensor<1x16x4x4xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       %0 = IE.MaxPool(%arg0)
    // CHECK-SAME:     kernel_size = [2 : i32, 2 : i32],
    // CHECK-SAME:     pads_begin = [0 : i32, 0 : i32],
    // CHECK-SAME:     pads_end = [0 : i32, 0 : i32],
    // CHECK-SAME:     post_op = {kind = "RELU", params = {}},
    // CHECK-SAME:     rounding_type = "CEIL",
    // CHECK-SAME:     strides = [1 : i32, 1 : i32]
    // CHECK-NOT:   IE.ReLU
}

// -----

func @DepthWiseConv2dWithReluTest(%arg0: tensor<1x16x4x4xf16>) -> tensor<1x16x3x3xf16> {
    %filters = const.Declare tensor<16x1x2x2xf16> = #const.Content<dense<1.0> : tensor<16x1x1x2x2xf16>, [#const.Reshape<[16, 1, 2, 2]>]>
    %0 = IE.GroupConvolution(%arg0, %filters)
        {
            dilations = [1 : i32, 1 : i32],
            groups = 16 : i32,
            strides = [1 : i32, 1 : i32],
            pads_begin = [0 : i32, 0 : i32],
            pads_end = [0 : i32, 0 : i32]
        } :
        tensor<1x16x4x4xf16>, tensor<16x1x2x2xf16> -> tensor<1x16x3x3xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x16x3x3xf16> -> tensor<1x16x3x3xf16>

    return %1 : tensor<1x16x3x3xf16>

    // CHECK:       %1 = IE.GroupConvolution(%arg0, %0)
    // CHECK-SAME:     dilations = [1 : i32, 1 : i32],
    // CHECK-SAME:     groups = 16 : i32,
    // CHECK-SAME:     pads_begin = [0 : i32, 0 : i32],
    // CHECK-SAME:     pads_end = [0 : i32, 0 : i32],
    // CHECK-SAME:     post_op = {kind = "RELU", params = {}},
    // CHECK-SAME:     strides = [1 : i32, 1 : i32]
    // CHECK-NOT:   IE.ReLU
}
