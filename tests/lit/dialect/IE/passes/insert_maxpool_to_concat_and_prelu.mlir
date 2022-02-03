// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB compilation-mode=DefaultHW" --insert-maxpool-to-concat-prelu %s | FileCheck %s

// CHECK-LABEL: @InsertMaxPoolToConcatAntPRelu
func @InsertMaxPoolToConcatAntPRelu(%arg0: tensor<1x1x30x30xf32>, %arg1: tensor<1x1x30x30xf32>) -> tensor<1x2x30x30xf32> {
    %0 = IE.Concat(%arg0, %arg1) {per_axis = {axis = 1}} : tensor<1x1x30x30xf32>, tensor<1x1x30x30xf32> -> tensor<1x2x30x30xf32>
    %neg_slope = const.Declare tensor<1x2xf32> = #const.Content<dense<[[0.1, 0.2]]> : tensor<1x2xf32>>
    %1 = IE.PRelu(%0, %neg_slope) : tensor<1x2x30x30xf32>, tensor<1x2xf32> -> tensor<1x2x30x30xf32>

    return %1 : tensor<1x2x30x30xf32>

    // CHECK:   %[[CST:.*]] = const.Declare tensor<1x2xf32> = #const.Content<dense<{{\[\[}}1.000000e-01, 2.000000e-01]]> : tensor<1x2xf32>>
    // CHECK:   %[[VAL_0:.*]] = IE.Concat(%arg0, %arg1) {per_axis = {axis = 1 : i64}} : tensor<1x1x30x30xf32>, tensor<1x1x30x30xf32> -> tensor<1x2x30x30xf32>
    // CHECK:   %[[VAL_1:.*]] = IE.MaxPool(%[[VAL_0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x2x30x30xf32> -> tensor<1x2x30x30xf32>
    // CHECK:   %[[VAL_2:.*]] = IE.PRelu(%[[VAL_1]], %[[CST]]) : tensor<1x2x30x30xf32>, tensor<1x2xf32> -> tensor<1x2x30x30xf32>
    // CHECK:   return %[[VAL_2]] : tensor<1x2x30x30xf32>
}
