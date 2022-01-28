// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=KMB compilation-mode=DefaultHW" --handle-large-strides --canonicalize %s | FileCheck %s


// CHECK-LABEL: @HandleLargeStridesPrimeStride
func @HandleLargeStridesPrimeStride(%arg0: tensor<1x16x28x28xf16>) -> tensor<1x32x3x3xf16> {
  %0 = const.Declare tensor<32x16x3x3xf16> = #const.Content<dense<1.0> : tensor<32x16x3x3xf16>>

  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [11, 11]} : tensor<1x16x28x28xf16>, tensor<32x16x3x3xf16> -> tensor<1x32x3x3xf16>

  // CHECK:       %[[CST:.*]] = const.Declare
  // CHECK:       %[[SLICED_INPUT0:.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV0:.*]] = IE.Convolution(%[[SLICED_INPUT0]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[SLICED_INPUT1:.*]] = IE.Slice %arg0 [0, 0, 0, 11] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV1:.*]] = IE.Convolution(%[[SLICED_INPUT1]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[SLICED_INPUT2:.*]] = IE.Slice %arg0 [0, 0, 0, 22] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV2:.*]] = IE.Convolution(%[[SLICED_INPUT2]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[CONCAT0:.*]] = IE.Concat(%[[SLICED_CONV0]], %[[SLICED_CONV1]], %[[SLICED_CONV2]])
  // CHECK-SAME{LITERAL}:       {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]}
  // CHECK-SAME:      : tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x1x3xf16>

  // CHECK:       %[[SLICED_INPUT3:.*]] = IE.Slice %arg0 [0, 0, 11, 0] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV3:.*]] = IE.Convolution(%[[SLICED_INPUT3]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[SLICED_INPUT4:.*]] = IE.Slice %arg0 [0, 0, 11, 11] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV4:.*]] = IE.Convolution(%[[SLICED_INPUT4]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[SLICED_INPUT5:.*]] = IE.Slice %arg0 [0, 0, 11, 22] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV5:.*]] = IE.Convolution(%[[SLICED_INPUT5]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[CONCAT1:.*]] = IE.Concat(%[[SLICED_CONV3]], %[[SLICED_CONV4]], %[[SLICED_CONV5]])
  // CHECK-SAME{LITERAL}:       {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]}
  // CHECK-SAME:      : tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x1x3xf16>

  // CHECK:       %[[SLICED_INPUT6:.*]] = IE.Slice %arg0 [0, 0, 22, 0] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV6:.*]] = IE.Convolution(%[[SLICED_INPUT6]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[SLICED_INPUT7:.*]] = IE.Slice %arg0 [0, 0, 22, 11] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV7:.*]] = IE.Convolution(%[[SLICED_INPUT7]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[SLICED_INPUT8:.*]] = IE.Slice %arg0 [0, 0, 22, 22] [1, 16, 3, 3] : tensor<1x16x28x28xf16> to tensor<1x16x3x3xf16>

  // CHECK:       %[[SLICED_CONV8:.*]] = IE.Convolution(%[[SLICED_INPUT8]], %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}

  // CHECK:       %[[CONCAT2:.*]] = IE.Concat(%[[SLICED_CONV6]], %[[SLICED_CONV7]], %[[SLICED_CONV8]])
  // CHECK-SAME{LITERAL}:       {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2]]}
  // CHECK-SAME:      : tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16>, tensor<1x32x1x1xf16> -> tensor<1x32x1x3xf16>

  // CHECK:       %[[CONCAT:.*]] = IE.Concat(%[[CONCAT0]], %[[CONCAT1]], %[[CONCAT2]])
  // CHECK-SAME{LITERAL}:       {static_offsets = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 0]]}
  // CHECK-SAME:      : tensor<1x32x1x3xf16>, tensor<1x32x1x3xf16>, tensor<1x32x1x3xf16> -> tensor<1x32x3x3xf16>

  return %1 : tensor<1x32x3x3xf16>
  // CHECK        return %[[CONCAT]]
}

// -----

// CHECK-LABEL: @HandleLargeStridesNonPrimeStride
func @HandleLargeStridesNonPrimeStride(%arg0: tensor<1x16x28x28xf16>) -> tensor<1x32x2x2xf16> {
  %0 = const.Declare tensor<32x16x11x11xf16> = #const.Content<dense<1.0> : tensor<32x16x11x11xf16>>

  %1 = IE.Convolution(%arg0, %0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [10, 10]} : tensor<1x16x28x28xf16>, tensor<32x16x11x11xf16> -> tensor<1x32x2x2xf16>

  // CHECK:       %[[CST:.*]] = const.Declare
  // CHECK:       %[[CONV0:.*]] = IE.Convolution(%arg0, %[[CST]])
  // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [5, 5]}

  // CHECK:       %[[MAXPOOL:.*]] = IE.MaxPool(%[[CONV0]])
  // CHECK-SAME:  {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [2, 2]} : tensor<1x32x4x4xf16> -> tensor<1x32x2x2xf16>

  return %1 : tensor<1x32x2x2xf16>
  // CHECK        return %[[MAXPOOL]]
}

// -----
