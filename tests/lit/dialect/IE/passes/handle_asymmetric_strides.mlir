// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB compilation-mode=DefaultHW" --handle-asymmetric-strides --canonicalize %s | FileCheck %s

// CHECK-LABEL: @HandleConvolutionWithAsymmetricStrides
func @HandleConvolutionWithAsymmetricStrides(%arg0: tensor<1x16x64x1024xf16>) -> tensor<1x32x64x512xf16> {
  %0 = const.Declare tensor<32x16x1x3xf16> = #const.Content<dense<1.0> : tensor<32x16x1x3xf16>>

  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 2]} : tensor<1x16x64x1024xf16>, tensor<32x16x1x3xf16> -> tensor<1x32x64x512xf16>

  return %1 : tensor<1x32x64x512xf16>

  // CHECK:       %[[CST:.*]] = const.Declare

  // CHECK:       %[[SLICED_CONV0:.*]] = IE.Convolution(%arg0, %[[CST]])
  // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [2, 2]}

  // CHECK:       %[[SLICED_INPUT1:.*]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 16, 63, 1024]

  // CHECK:       %[[SLICED_CONV1:.*]] = IE.Convolution(%[[SLICED_INPUT1]], %[[CST]])
  // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]}
  // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[SLICED_CONV0]], %[[SLICED_CONV1]])
  // CHECK-SAME:    {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}}

  // CHECK        return %[[CONCAT]]
}

// -----

// CHECK-LABEL: func @ConvWithAsymStridesEqualInputAndKernel
func @ConvWithAsymStridesEqualInputAndKernel(%arg0: tensor<1x256x4x1xf16>) -> tensor<1x256x2x1xf16> {
  %0 = const.Declare tensor<256x256x3x1xf16> = #const.Content<dense<1.0> : tensor<256x256x3x1xf16>>

  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 0], strides = [2, 1]} : tensor<1x256x4x1xf16>, tensor<256x256x3x1xf16> -> tensor<1x256x2x1xf16>

  return %1 : tensor<1x256x2x1xf16>

  // CHECK:       %[[CST:.*]] = const.Declare

  // CHECK:       %[[CONV:.*]] = IE.Convolution(%arg0, %[[CST]])
  // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 0], strides = [2, 2]}

  // CHECK        return %[[CONV]]
}

// -----

// CHECK-LABEL: @HandleConvolutionWithAsymmetricStridesWithFQ
func @HandleConvolutionWithAsymmetricStridesWithFQ(%arg0: tensor<1x16x64x1024xf16>) -> tensor<1x32x64x512xf16> {
  %0 = const.Declare tensor<32x16x1x3xf16> = #const.Content<dense<1.0> : tensor<32x16x1x3xf16>>
  %1 = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<0.0> : tensor<1x1x1x1xf16>>
  %2 = const.Declare tensor<1x1x1x1xf16> = #const.Content<dense<1.0> : tensor<1x1x1x1xf16>>
  %3 = const.Declare tensor<32x1x1x1xf16> = #const.Content<dense<0.0> : tensor<32x1x1x1xf16>>
  %4 = const.Declare tensor<32x1x1x1xf16> = #const.Content<dense<1.0> : tensor<32x1x1x1xf16>>

  %5 = IE.FakeQuantize(%arg0, %1, %2, %1, %2) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x16x64x1024xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x16x64x1024xf16>
  %6 = IE.FakeQuantize(%0, %1, %2, %3, %4) {auto_broadcast = "NUMPY", levels = 255 : i64} : tensor<32x16x1x3xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<32x1x1x1xf16>, tensor<32x1x1x1xf16> -> tensor<32x16x1x3xf16>
  %7 = IE.Convolution(%5, %6) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 2]} : tensor<1x16x64x1024xf16>, tensor<32x16x1x3xf16> -> tensor<1x32x64x512xf16>
  %8 = IE.FakeQuantize(%7, %1, %2, %1, %2) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x32x64x512xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16>, tensor<1x1x1x1xf16> -> tensor<1x32x64x512xf16>

  return %8 : tensor<1x32x64x512xf16>

  // CHECK:       %[[CST0:.*]] = const.Declare
  // CHECK:       %[[CST1:.*]] = const.Declare
  // CHECK:       %[[CST2:.*]] = const.Declare
  // CHECK:       %[[CST3:.*]] = const.Declare
  // CHECK:       %[[CST4:.*]] = const.Declare

  // CHECK:       %[[VAR5:.*]] = IE.FakeQuantize
  // CHECK:       %[[VAR6:.*]] = IE.FakeQuantize

  // CHECK:       %[[SLICED_CONV0:.*]] = IE.Convolution(%[[VAR5]], %[[VAR6]])
  // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [2, 2]}

  // CHECK:       %[[SLICED_FQ0:.*]] = IE.FakeQuantize(%[[SLICED_CONV0]], %[[CST3]], %[[CST2]], %[[CST3]], %[[CST2]])

  // CHECK:       %[[SLICED_INPUT1:.*]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 16, 63, 1024]

  // CHECK:       %[[SLICED_FQ:.*]] = IE.FakeQuantize(%[[SLICED_INPUT1]], %[[CST3]], %[[CST2]], %[[CST3]], %[[CST2]])

  // CHECK:       %[[SLICED_CONV1:.*]] = IE.Convolution(%[[SLICED_FQ]], %[[VAR6]])
  // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [2, 2]}

  // CHECK:       %[[SLICED_FQ1:.*]] = IE.FakeQuantize(%[[SLICED_CONV1]], %[[CST3]], %[[CST2]], %[[CST3]], %[[CST2]])

  // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[SLICED_FQ0]], %[[SLICED_FQ1]])
  // CHECK-SAME:    {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}}

  // CHECK:       %[[OUT_FQ:.*]] = IE.FakeQuantize(%[[CONCAT]], %[[CST3]], %[[CST2]], %[[CST3]], %[[CST2]])

  // CHECK        return %[[OUT_FQ]]
}

// -----

// CHECK-LABEL: @HandleAvgPoolWithAsymmetricStrides
func @HandleAvgPoolWithAsymmetricStrides(%arg0 : tensor<1x16x30x30xf16>) -> tensor<1x16x28x14xf16> {
    %0 = IE.AvgPool(%arg0) {
            exclude_pads,
            kernel_size = [3, 3],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            rounding_type = "FLOOR",
            strides = [1, 2]
    } : tensor<1x16x30x30xf16> -> tensor<1x16x28x14xf16>

    return %0 : tensor<1x16x28x14xf16>

    // CHECK:       %[[VAL0:.*]] = IE.AvgPool(%arg0)
    // CHECK-SAME:    {exclude_pads, kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [2, 2]} :
    // CHECK-SAME:    tensor<1x16x30x30xf16> -> tensor<1x16x14x14xf16>

    // CHECK:       %[[VAL1:.*]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 16, 29, 30] : tensor<1x16x30x30xf16> to tensor<1x16x29x30xf16>

    // CHECK:       %[[VAL2:.*]] = IE.AvgPool(%[[VAL1]])
    // CHECK-SAME:    {exclude_pads, kernel_size = [3, 3], pads_begin = [0, 0], pads_end = [1, 0], rounding_type = "FLOOR", strides = [2, 2]} :
    // CHECK-SAME:    tensor<1x16x29x30xf16> -> tensor<1x16x14x14xf16>

    // CHECK:       %[[VAL3:.*]] = IE.Concat(%[[VAL0]], %[[VAL2]])
    // CHECK-SAME:    {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} :
    // CHECK-SAME:    tensor<1x16x14x14xf16>, tensor<1x16x14x14xf16> -> tensor<1x16x28x14xf16>

    // CHECK:       return %[[VAL3]] : tensor<1x16x28x14xf16>
}

// -----

// CHECK-LABEL: @HandleAvgPoolWithEqualInputAndKernel
func @HandleAvgPoolWithEqualInputAndKernel(%arg0 : tensor<1x16x7x7xf16>) -> tensor<1x16x1x1xf16> {
    %0 = IE.AvgPool(%arg0) {
              exclude_pads,
              kernel_size = [7, 7],
              pads_begin = [0, 0],
              pads_end = [0, 0],
              rounding_type = "FLOOR",
              strides = [1, 2]
        } : tensor<1x16x7x7xf16> -> tensor<1x16x1x1xf16>

    return %0 : tensor<1x16x1x1xf16>

    // CHECK:       %[[VAL0:.*]] = IE.AvgPool(%arg0)
    // CHECK-SAME:    {exclude_pads, kernel_size = [7, 7], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [2, 2]} :
    // CHECK-SAME:    tensor<1x16x7x7xf16> -> tensor<1x16x1x1xf16>

    // CHECK:       return %[[VAL0]] : tensor<1x16x1x1xf16>
}
