// RUN: vpux-opt --split-input-file --expand-activation-channels --canonicalize %s | FileCheck %s

// CHECK-LABEL: @ExpandMaxPoolChannels
func @ExpandMaxPoolChannels(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x15x13xf16> {
  %0 = IE.MaxPool(%arg0) {kernel_size = [5 : i32, 5 : i32], pads_begin = [2 : i32, 0 : i32], pads_end = [2 : i32, 0 : i32], rounding_type = "FLOOR", strides = [2 : i32, 2 : i32]} : tensor<1x3x30x30xf16> -> tensor<1x3x15x13xf16>
  // CHECK:       %[[PAD:.*]] = IE.Expand(%arg0) {pads_begin_attr = [0 : i32, 0 : i32, 0 : i32, 0 : i32], pads_end_attr = [0 : i32, 5 : i32, 0 : i32, 0 : i32]} : tensor<1x3x30x30xf16> -> tensor<1x8x30x30xf16>
  // CHECK:       %[[POOL:.*]] = IE.MaxPool(%[[PAD]]) {kernel_size = [5 : i32, 5 : i32], pads_begin = [2 : i32, 0 : i32], pads_end = [2 : i32, 0 : i32], rounding_type = "FLOOR", strides = [2 : i32, 2 : i32]} : tensor<1x8x30x30xf16> -> tensor<1x8x15x13xf16>
  // CHECK:       %[[OUT:.*]] = subtensor %[[POOL]][0, 0, 0, 0] [1, 3, 15, 13] [1, 1, 1, 1] : tensor<1x8x15x13xf16> to tensor<1x3x15x13xf16>

  return %0 : tensor<1x3x15x13xf16>
  // CHECK        return %[[OUT]]
}

// CHECK-LABEL: @ExpandAvgPoolChannels
func @ExpandAvgPoolChannels(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x16x16xf16> {
  %0 = IE.AvgPool(%arg0) {kernel_size = [5 : i32, 5 : i32], pads_begin = [2 : i32, 2 : i32], pads_end = [3 : i32, 3 : i32], rounding_type = "CEIL", strides = [2 : i32, 2 : i32]} : tensor<1x3x30x30xf16> -> tensor<1x3x16x16xf16>
  // CHECK:       %[[PAD:.*]] = IE.Expand(%arg0) {pads_begin_attr = [0 : i32, 0 : i32, 0 : i32, 0 : i32], pads_end_attr = [0 : i32, 5 : i32, 0 : i32, 0 : i32]} : tensor<1x3x30x30xf16> -> tensor<1x8x30x30xf16>
  // CHECK:       %[[POOL:.*]] = IE.AvgPool(%[[PAD]]) {kernel_size = [5 : i32, 5 : i32], pads_begin = [2 : i32, 2 : i32], pads_end = [3 : i32, 3 : i32], rounding_type = "CEIL", strides = [2 : i32, 2 : i32]} : tensor<1x8x30x30xf16> -> tensor<1x8x16x16xf16>
  // CHECK:       %[[OUT:.*]] = subtensor %[[POOL]][0, 0, 0, 0] [1, 3, 16, 16] [1, 1, 1, 1] : tensor<1x8x16x16xf16> to tensor<1x3x16x16xf16>

  return %0 : tensor<1x3x16x16xf16>
  // CHECK        return %[[OUT]]
}

// CHECK-LABEL: @ExpandConvolutionChannels
func @ExpandConvolutionChannels(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x5x24x9xf16> {
  %0 = const.Declare tensor<5x3x3x5xf16> = #const.Content<dense<1.0> : tensor<5x3x3x5xf16>>

  // CHECK:       %[[EXTENDED_FILTER:.*]] = const.Declare tensor<8x8x3x5xf16> =
  // CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<5x3x3x5xf16>, [#const.PadWithZero<[0, 0, 0, 0], [3, 5, 0, 0]>]>
  // CHECK:       %[[EXTENDED_INPUT:.*]] = IE.Expand(%arg0)

  %1 = IE.Convolution(%arg0, %0) {dilations = [3 : i32, 1 : i32], pads_begin = [0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32], strides = [1 : i32, 3 : i32]} : tensor<1x3x30x30xf16>, tensor<5x3x3x5xf16> -> tensor<1x5x24x9xf16>
  // CHECK:       %[[EXTENDED_CONV:.*]] = IE.Convolution(%[[EXTENDED_INPUT]], %[[EXTENDED_FILTER]])
  // CHECK:       %[[REDUNDRANT_SUBTENSOR:.*]] = subtensor %[[EXTENDED_CONV]]

  return %1 : tensor<1x5x24x9xf16>
  // CHECK        return %[[REDUNDRANT_SUBTENSOR]]
}

// CHECK-LABEL: @ExpandBiasesConvolutionChannels
func @ExpandBiasesConvolutionChannels(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x5x24x9xf16> {
  %0 = const.Declare tensor<5x3x3x5xf16> = #const.Content<dense<1.0> : tensor<5x3x3x5xf16>>
  %1 = const.Declare tensor<1x5x1x1xf16> = #const.Content<dense<1.0> : tensor<1x5x1x1xf16>>

  // CHECK-DAG:   %[[EXTENDED_FILTER:.*]] = const.Declare tensor<8x8x3x5xf16> = #const.Content<dense<1.000000e+00> : tensor<5x3x3x5xf16>, [#const.PadWithZero<[0, 0, 0, 0], [3, 5, 0, 0]>]>
  // CHECK-DAG:   %[[EXTENDED_BIAS:.*]] = const.Declare tensor<1x8x1x1xf16> = #const.Content<dense<1.000000e+00> : tensor<1x5x1x1xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 3, 0, 0]>]>

  // CHECK:       %[[EXTENDED_INPUT:.*]] = IE.Expand(%arg0)

  %2 = IE.Convolution(%arg0, %0, %1) {dilations = [3 : i32, 1 : i32], pads_begin = [0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32], strides = [1 : i32, 3 : i32]} : tensor<1x3x30x30xf16>, tensor<5x3x3x5xf16>, tensor<1x5x1x1xf16> -> tensor<1x5x24x9xf16>
  // CHECK:       %[[EXTENDED_CONV:.*]] = IE.Convolution(%[[EXTENDED_INPUT]], %[[EXTENDED_FILTER]], %[[EXTENDED_BIAS]])
  // CHECK:       %[[REDUNDRANT_SUBTENSOR:.*]] = subtensor %[[EXTENDED_CONV]]

  return %2 : tensor<1x5x24x9xf16>
  // CHECK        return %[[REDUNDRANT_SUBTENSOR]]
}
