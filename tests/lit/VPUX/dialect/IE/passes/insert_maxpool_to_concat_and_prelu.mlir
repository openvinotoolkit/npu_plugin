// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --insert-maxpool-to-concat-lrelu %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @InsertMaxPoolToConcatAntLRelu
func @InsertMaxPoolToConcatAntLRelu(%arg0: tensor<1x128x2x32xf16>, %arg1: tensor<1x128x1x32xf16>) -> tensor<1x128x3x32xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x128x2x32xf16>, tensor<1x128x1x32xf16> -> tensor<1x128x3x32xf16>
    %1 = IE.LeakyRelu(%0) {negative_slope = 0.000000e+00 : f64} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>

    return %1 : tensor<1x128x3x32xf16>

    // CHECK:   %[[VAL_0:.*]] = IE.Concat(%arg0, %arg1) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x128x2x32xf16>, tensor<1x128x1x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   %[[VAL_1:.*]] = IE.MaxPool(%[[VAL_0]]) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   %[[VAL_2:.*]] = IE.LeakyRelu(%[[VAL_1]]) {negative_slope = 0.000000e+00 : f64} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   return %[[VAL_2]] : tensor<1x128x3x32xf16>
}
