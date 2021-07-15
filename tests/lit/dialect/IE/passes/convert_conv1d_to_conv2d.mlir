// RUN: vpux-opt --split-input-file --convert-conv1d-to-conv2d %s | FileCheck %s

// CHECK-LABEL: @ConvertConv1DToConv2D
func @ConvertConv1DToConv2D(%arg0: tensor<1x16x64xf16>) -> tensor<1x1x61xf16> {
    %FILTERS = const.Declare tensor<1x16x5xf16> = #const.Content<dense<1.000000e+00> : tensor<1x16x5xf16>>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {dilations = [2 : i32], pads_begin = [3 : i32], pads_end = [2 : i32], strides = [1 : i32]} : tensor<1x16x64xf16>, tensor<1x16x5xf16> -> tensor<1x1x61xf16>
    return %RESULT : tensor<1x1x61xf16>

    // CHECK:       %[[VAL0:.*]] = IE.Convolution
    // CHECK-SAME:      dilations = [1 : i32, 2 : i32]
    // CHECK-SAME:      pads_begin = [0 : i32, 3 : i32]
    // CHECK-SAME:      pads_end = [0 : i32, 2 : i32]
    // CHECK-SAME:      strides = [1 : i32, 1 : i32]
    // CHECK-SAME:      tensor<1x16x1x64xf16>, tensor<1x16x1x5xf16> -> tensor<1x1x1x61xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 1, 61]} : tensor<1x1x1x61xf16> -> tensor<1x1x61xf16>
    // CHECK:       return %[[RESULT]]
}

// CHECK-LABEL: @ConvertConv1DToConv2DGroupConvolution
func @ConvertConv1DToConv2DGroupConvolution(%arg0: tensor<1x16x30xf16>) -> tensor<1x8x28xf16>{
    %FILTERS = const.Declare tensor<8x8x3xf16> = #const.Content<dense<1.000000e+00> : tensor<2x4x8x3xf32>, [#const.Reshape<[8, 8, 3]>, #const.ConvertElemType<f16>]>
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [1 : i32], groups = 2 : i32, pads_begin = [0 : i32], pads_end = [0 : i32], strides = [1 : i32]} : tensor<1x16x30xf16>, tensor<8x8x3xf16> -> tensor<1x8x28xf16>
    return %RESULT : tensor<1x8x28xf16>

    // CHECK:       %[[VAL0:.*]] = IE.GroupConvolution
    // CHECK-SAME:      dilations = [1 : i32, 1 : i32]
    // CHECK-SAME:      groups = 2 : i32
    // CHECK-SAME:      pads_begin = [0 : i32, 0 : i32]
    // CHECK-SAME:      pads_end = [0 : i32, 0 : i32]
    // CHECK-SAME:      strides = [1 : i32, 1 : i32]
    // CHECK-SAME:      tensor<1x16x1x30xf16>, tensor<8x8x1x3xf16> -> tensor<1x8x1x28xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 8, 28]} : tensor<1x8x1x28xf16> -> tensor<1x8x28xf16>
    // CHECK:       return %[[RESULT]]
}
