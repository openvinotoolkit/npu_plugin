// RUN: vpux-opt --split-input-file --support-batch-for-pad %s | FileCheck %s

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x10x20x30xf16>)
func @main(%arg0: tensor<1x10x20x30xf16>) -> tensor<5x10x20x30xf16> {
  %0 = IE.Pad(%arg0) {mode = "CONSTANT", pad_value_attr = 3.000000e+00 : f64, pads_begin_attr = [0, 0, 0, 0], pads_end_attr = [4, 0, 0, 0]} : tensor<1x10x20x30xf16> -> tensor<5x10x20x30xf16>
  return %0 : tensor<5x10x20x30xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Pad([[ARG0]]
    // CHECK-SAME:      mode = "CONSTANT"
    // CHECK-SAME:      pad_value_attr = 3.000000e+00
    // CHECK-SAME:      pads_begin_attr = [0, 0, 0, 0]
    // CHECK-SAME:      pads_end_attr = [0, 0, 0, 0]
    // CHECK:       [[CST:%.+]] = const.Declare tensor<4x10x20x30xf16> = #const.Content<dense<3.000000e+00> : tensor<4x10x20x30xf16>>
    // CHECK:       [[VAR1:%.+]] = IE.Concat([[VAR0]], [[CST]])
    // CHECK-SAME:      per_axis = {axis = 0 : i64}
    // CHECK:       return [[VAR1]] : tensor<5x10x20x30xf16>
}

// -----

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x10x20x30xf16>)
func @main(%arg0: tensor<1x10x20x30xf16>) -> tensor<6x15x25x35xf16> {
  %0 = IE.Pad(%arg0) {mode = "CONSTANT", pad_value_attr = 5.000000e+00 : f64, pads_begin_attr = [5, 4, 3, 2], pads_end_attr = [0, 1, 2, 3]} : tensor<1x10x20x30xf16> -> tensor<6x15x25x35xf16>
  return %0 : tensor<6x15x25x35xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Pad([[ARG0]]
    // CHECK-SAME:      mode = "CONSTANT"
    // CHECK-SAME:      pad_value_attr = 5.000000e+00
    // CHECK-SAME:      pads_begin_attr = [0, 4, 3, 2]
    // CHECK-SAME:      pads_end_attr = [0, 1, 2, 3]
    // CHECK:       [[CST:%.+]] = const.Declare tensor<5x15x25x35xf16> = #const.Content<dense<5.000000e+00> : tensor<5x15x25x35xf16>>
    // CHECK:       [[VAR1:%.+]] = IE.Concat([[CST]], [[VAR0]])
    // CHECK-SAME:      per_axis = {axis = 0 : i64}
    // CHECK:       return [[VAR1]] : tensor<6x15x25x35xf16>
}

// -----

// CHECK: func @main([[ARG0:%arg[0-9]+]]: tensor<1x10x20x30xf16>)
func @main(%arg0: tensor<1x10x20x30xf16>) -> tensor<6x15x25x35xf16> {
  %0 = IE.Pad(%arg0) {mode = "CONSTANT", pad_value_attr = 0.000000e+00 : f64, pads_begin_attr = [3, 3, 2, 2], pads_end_attr = [2, 2, 3, 3]} : tensor<1x10x20x30xf16> -> tensor<6x15x25x35xf16>
  return %0 : tensor<6x15x25x35xf16>

    // CHECK:       [[VAR0:%.+]] = IE.Pad([[ARG0]]
    // CHECK-SAME:      mode = "CONSTANT"
    // CHECK-SAME:      pad_value_attr = 0.000000e+00
    // CHECK-SAME:      pads_begin_attr = [0, 3, 2, 2]
    // CHECK-SAME:      pads_end_attr = [0, 2, 3, 3]
    // CHECK:       [[CST0:%.+]] = const.Declare tensor<3x15x25x35xf16> = #const.Content<dense<0.000000e+00> : tensor<3x15x25x35xf16>>
    // CHECK:       [[CST1:%.+]] = const.Declare tensor<2x15x25x35xf16> = #const.Content<dense<0.000000e+00> : tensor<2x15x25x35xf16>>
    // CHECK:       [[VAR1:%.+]] = IE.Concat([[CST0]], [[VAR0]], [[CST1]])
    // CHECK-SAME:      per_axis = {axis = 0 : i64}
    // CHECK:       return [[VAR1]] : tensor<6x15x25x35xf16>
}
