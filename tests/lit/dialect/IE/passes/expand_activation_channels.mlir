// RUN: vpux-opt --split-input-file --set-compile-params="vpu-arch=VPU3700 compilation-mode=ReferenceHW" --expand-activation-channels --canonicalize %s | FileCheck %s

// CHECK-LABEL: @ExpandMaxPoolChannels
func @ExpandMaxPoolChannels(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x3x15x13xf16> {
  %0 = IE.MaxPool(%arg0) {kernel_size = [5 : i32, 5 : i32], pads_begin = [2 : i32, 0 : i32], pads_end = [2 : i32, 0 : i32], rounding_type = "FLOOR", strides = [2 : i32, 2 : i32]} : tensor<1x3x30x30xf16> -> tensor<1x3x15x13xf16>
  // CHECK:       %[[PAD:.*]] = IE.Expand(%arg0) {pads_begin_attr = [0 : i32, 0 : i32, 0 : i32, 0 : i32], pads_end_attr = [0 : i32, 13 : i32, 0 : i32, 0 : i32]} : tensor<1x3x30x30xf16> -> tensor<1x16x30x30xf16>
  // CHECK:       %[[POOL:.*]] = IE.MaxPool(%[[PAD]]) {kernel_size = [5 : i32, 5 : i32], pads_begin = [2 : i32, 0 : i32], pads_end = [2 : i32, 0 : i32], rounding_type = "FLOOR", strides = [2 : i32, 2 : i32]} : tensor<1x16x30x30xf16> -> tensor<1x16x15x13xf16>
  // CHECK:       %[[OUT:.*]] = tensor.extract_slice %[[POOL]][0, 0, 0, 0] [1, 3, 15, 13] [1, 1, 1, 1] : tensor<1x16x15x13xf16> to tensor<1x3x15x13xf16>

  return %0 : tensor<1x3x15x13xf16>
  // CHECK        return %[[OUT]]
}

// CHECK-LABEL: @ExpandConvolutionChannels
func @ExpandConvolutionChannels(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x5x28x28xf16> {
  %0 = const.Declare tensor<5x3x3x3xf16> = #const.Content<dense<1.0> : tensor<5x3x3x3xf16>>

  // CHECK:       %[[EXTENDED_FILTER:.*]] = const.Declare tensor<16x16x3x3xf16> =
  // CHECK-SAME:      #const.Content<dense<1.000000e+00> : tensor<5x3x3x3xf16>, [#const.PadWithZero<[0, 0, 0, 0], [11, 13, 0, 0]>]>
  // CHECK:       %[[EXTENDED_INPUT:.*]] = IE.Expand(%arg0)

  %1 = IE.Convolution(%arg0, %0) {dilations = [1 : i32, 1 : i32], pads_begin = [0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : tensor<1x3x30x30xf16>, tensor<5x3x3x3xf16> -> tensor<1x5x28x28xf16>
  // CHECK:       %[[EXTENDED_CONV:.*]] = IE.Convolution(%[[EXTENDED_INPUT]], %[[EXTENDED_FILTER]])
  // CHECK:       %[[REDUNDRANT_SUBTENSOR:.*]] = tensor.extract_slice %[[EXTENDED_CONV]]

  return %1 : tensor<1x5x28x28xf16>
  // CHECK        return %[[REDUNDRANT_SUBTENSOR]]
}

// CHECK-LABEL: @ExpandBiasesConvolutionChannels
func @ExpandBiasesConvolutionChannels(%arg0: tensor<1x3x30x30xf16>) -> tensor<1x5x28x28xf16> {
  %0 = const.Declare tensor<5x3x3x3xf16> = #const.Content<dense<1.0> : tensor<5x3x3x3xf16>>
  %1 = const.Declare tensor<1x5x1x1xf16> = #const.Content<dense<1.0> : tensor<1x5x1x1xf16>>

  // CHECK-DAG:   %[[EXTENDED_FILTER:.*]] = const.Declare tensor<16x16x3x3xf16> = #const.Content<dense<1.000000e+00> : tensor<5x3x3x3xf16>, [#const.PadWithZero<[0, 0, 0, 0], [11, 13, 0, 0]>]>
  // CHECK-DAG:   %[[EXTENDED_BIAS:.*]] = const.Declare tensor<1x16x1x1xf16> = #const.Content<dense<1.000000e+00> : tensor<1x5x1x1xf16>, [#const.PadWithZero<[0, 0, 0, 0], [0, 11, 0, 0]>]>

  // CHECK:       %[[EXTENDED_INPUT:.*]] = IE.Expand(%arg0)

  %2 = IE.Convolution(%arg0, %0, %1) {dilations = [1 : i32, 1 : i32], pads_begin = [0 : i32, 0 : i32], pads_end = [0 : i32, 0 : i32], strides = [1 : i32, 1 : i32]} : tensor<1x3x30x30xf16>, tensor<5x3x3x3xf16>, tensor<1x5x1x1xf16> -> tensor<1x5x28x28xf16>
  // CHECK:       %[[EXTENDED_CONV:.*]] = IE.Convolution(%[[EXTENDED_INPUT]], %[[EXTENDED_FILTER]], %[[EXTENDED_BIAS]])
  // CHECK:       %[[REDUNDRANT_SUBTENSOR:.*]] = tensor.extract_slice %[[EXTENDED_CONV]]

  return %2 : tensor<1x5x28x28xf16>
  // CHECK        return %[[REDUNDRANT_SUBTENSOR]]
}

// CHECK-LABEL: @ExpandEltwiseAddChannels
func @ExpandEltwiseAddChannels(%arg0: tensor<1x3x30x25xf16>, %arg1: tensor<1x3x30x25xf16>) -> tensor<1x3x30x25xf16> {
  %0 = IE.Add(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x3x30x25xf16>, tensor<1x3x30x25xf16> -> tensor<1x3x30x25xf16>
  // CHECK:       %[[EXPAND_LEFT_INPUT:.*]] = IE.Expand(%arg0) {pads_begin_attr = [0 : i32, 0 : i32, 0 : i32, 0 : i32], pads_end_attr = [0 : i32, 13 : i32, 0 : i32, 0 : i32]} : tensor<1x3x30x25xf16> -> tensor<1x16x30x25xf16>
  // CHECK:       %[[EXPAND_RIGHT_INPUT:.*]] = IE.Expand(%arg1) {pads_begin_attr = [0 : i32, 0 : i32, 0 : i32, 0 : i32], pads_end_attr = [0 : i32, 13 : i32, 0 : i32, 0 : i32]} : tensor<1x3x30x25xf16> -> tensor<1x16x30x25xf16>
  // CHECK:       %[[ELTWISE_ADD:.*]] = IE.Add(%[[EXPAND_LEFT_INPUT]], %[[EXPAND_RIGHT_INPUT]]) {auto_broadcast = "NUMPY"} : tensor<1x16x30x25xf16>, tensor<1x16x30x25xf16> -> tensor<1x16x30x25xf16>
  // CHECK:       %[[OUT:.*]] = tensor.extract_slice %[[ELTWISE_ADD]][0, 0, 0, 0] [1, 3, 30, 25] [1, 1, 1, 1] : tensor<1x16x30x25xf16> to tensor<1x3x30x25xf16>

  return %0 : tensor<1x3x30x25xf16>
  // CHECK        return %[[OUT]]
}
