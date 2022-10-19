// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --legalize-dilated-conv %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertDilatedConvolutionToConvolution1
func @ConvertDilatedConvolutionToConvolution1(%arg0: tensor<1x64x20x20xf16>) -> tensor<1x64x18x2xf16> {
    %FILTERS = const.Declare tensor<64x64x3x3xf16> = #const.Content<dense<1.000000e+00> : tensor<64x64x3x3xf16>>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 9], output_padding = [0, 0]} : tensor<1x64x20x20xf16>, tensor<64x64x3x3xf16> -> tensor<1x64x18x2xf16>
    return %RESULT : tensor<1x64x18x2xf16>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<64x64x3x3xf16> = #const.Content<dense<1.000000e+00> : tensor<64x64x3x3xf16>>
    // CHECK:       %[[SLICED_FILTER0:.*]] = IE.Slice %[[CST]] [0, 0, 0, 0] [64, 64, 3, 1] : tensor<64x64x3x3xf16> to tensor<64x64x3x1xf16>
    // CHECK:       %[[SLICED_FILTER1:.*]] = IE.Slice %[[CST]] [0, 0, 0, 1] [64, 64, 3, 1] : tensor<64x64x3x3xf16> to tensor<64x64x3x1xf16>
    // CHECK:       %[[SLICED_FILTER2:.*]] = IE.Slice %[[CST]] [0, 0, 0, 2] [64, 64, 3, 1] : tensor<64x64x3x3xf16> to tensor<64x64x3x1xf16>
    // CHECK:       %[[SLICED_INPUT0:.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 20, 1] : tensor<1x64x20x20xf16> to tensor<1x64x20x1xf16>
    // CHECK:       %[[CONV0:.*]] = IE.Convolution(%[[SLICED_INPUT0]], %[[SLICED_FILTER0]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x64x20x1xf16>, tensor<64x64x3x1xf16> -> tensor<1x64x18x1xf16>

    // CHECK:       %[[SLICED_INPUT1:.*]] = IE.Slice %arg0 [0, 0, 0, 9] [1, 64, 20, 1] : tensor<1x64x20x20xf16> to tensor<1x64x20x1xf16>
    // CHECK:       %[[CONV1:.*]] = IE.Convolution(%[[SLICED_INPUT1]], %[[SLICED_FILTER1]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x64x20x1xf16>, tensor<64x64x3x1xf16> -> tensor<1x64x18x1xf16>
    // CHECK:       %[[ADD0:.*]] = IE.Add(%[[CONV0]], %[[CONV1]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x64x18x1xf16>, tensor<1x64x18x1xf16> -> tensor<1x64x18x1xf16>

    // CHECK:       %[[SLICED_INPUT2:.*]] = IE.Slice %arg0 [0, 0, 0, 18] [1, 64, 20, 1] : tensor<1x64x20x20xf16> to tensor<1x64x20x1xf16>
    // CHECK:       %[[CONV2:.*]] = IE.Convolution(%[[SLICED_INPUT2]], %[[SLICED_FILTER2]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x64x20x1xf16>, tensor<64x64x3x1xf16> -> tensor<1x64x18x1xf16>

    // CHECK:       %[[ADD1:.*]] = IE.Add(%[[ADD0]], %[[CONV2]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x64x18x1xf16>, tensor<1x64x18x1xf16> -> tensor<1x64x18x1xf16>

    // CHECK:       %[[SLICED_INPUT3:.*]] = IE.Slice %arg0 [0, 0, 0, 1] [1, 64, 20, 1] : tensor<1x64x20x20xf16> to tensor<1x64x20x1xf16>
    // CHECK:       %[[CONV3:.*]] = IE.Convolution(%[[SLICED_INPUT3]], %[[SLICED_FILTER0]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x64x20x1xf16>, tensor<64x64x3x1xf16> -> tensor<1x64x18x1xf16>

    // CHECK:       %[[SLICED_INPUT4:.*]] = IE.Slice %arg0 [0, 0, 0, 10] [1, 64, 20, 1] : tensor<1x64x20x20xf16> to tensor<1x64x20x1xf16>
    // CHECK:       %[[CONV4:.*]] = IE.Convolution(%[[SLICED_INPUT4]], %[[SLICED_FILTER1]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x64x20x1xf16>, tensor<64x64x3x1xf16> -> tensor<1x64x18x1xf16>
    // CHECK:       %[[ADD2:.*]] = IE.Add(%[[CONV3]], %[[CONV4]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x64x18x1xf16>, tensor<1x64x18x1xf16> -> tensor<1x64x18x1xf16>

    // CHECK:       %[[SLICED_INPUT5:.*]] = IE.Slice %arg0 [0, 0, 0, 19] [1, 64, 20, 1] : tensor<1x64x20x20xf16> to tensor<1x64x20x1xf16>
    // CHECK:       %[[CONV5:.*]] = IE.Convolution(%[[SLICED_INPUT5]], %[[SLICED_FILTER2]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x64x20x1xf16>, tensor<64x64x3x1xf16> -> tensor<1x64x18x1xf16>

    // CHECK:       %[[ADD3:.*]] = IE.Add(%[[ADD2]], %[[CONV5]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x64x18x1xf16>, tensor<1x64x18x1xf16> -> tensor<1x64x18x1xf16>

    // CHECK:       %[[CONCAT:.*]]= IE.Concat(%[[ADD1]], %[[ADD3]]) {per_axis = {axis = 3 : i64}} : tensor<1x64x18x1xf16>, tensor<1x64x18x1xf16> -> tensor<1x64x18x2xf16>
    // CHECK:       return %[[CONCAT]]

}

// CHECK-LABEL: @ConvertDilatedConvolutionToConvolution2
func @ConvertDilatedConvolutionToConvolution2(%arg0: tensor<1x64x20x20xf16>) -> tensor<1x64x2x18xf16> {
    %FILTERS = const.Declare tensor<64x64x3x3xf16> = #const.Content<dense<1.000000e+00> : tensor<64x64x3x3xf16>>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [9, 1], output_padding = [0, 0]} : tensor<1x64x20x20xf16>, tensor<64x64x3x3xf16> -> tensor<1x64x2x18xf16>
    return %RESULT : tensor<1x64x2x18xf16>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<64x64x3x3xf16> = #const.Content<dense<1.000000e+00> : tensor<64x64x3x3xf16>>
    // CHECK:       %[[SLICED_FILTER0:.*]] = IE.Slice %[[CST]] [0, 0, 0, 0] [64, 64, 1, 3] : tensor<64x64x3x3xf16> to tensor<64x64x1x3xf16>
    // CHECK:       %[[SLICED_FILTER1:.*]] = IE.Slice %[[CST]] [0, 0, 1, 0] [64, 64, 1, 3] : tensor<64x64x3x3xf16> to tensor<64x64x1x3xf16>
    // CHECK:       %[[SLICED_FILTER2:.*]] = IE.Slice %[[CST]] [0, 0, 2, 0] [64, 64, 1, 3] : tensor<64x64x3x3xf16> to tensor<64x64x1x3xf16>
    // CHECK:       %[[SLICED_INPUT0:.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 64, 1, 20] : tensor<1x64x20x20xf16> to tensor<1x64x1x20xf16>
    // CHECK:       %[[CONV0:.*]] = IE.Convolution(%[[SLICED_INPUT0]], %[[SLICED_FILTER0]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x64x1x20xf16>, tensor<64x64x1x3xf16> -> tensor<1x64x1x18xf16>

    // CHECK:       %[[SLICED_INPUT1:.*]] = IE.Slice %arg0 [0, 0, 9, 0] [1, 64, 1, 20] : tensor<1x64x20x20xf16> to tensor<1x64x1x20xf16>
    // CHECK:       %[[CONV1:.*]] = IE.Convolution(%[[SLICED_INPUT1]], %[[SLICED_FILTER1]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x64x1x20xf16>, tensor<64x64x1x3xf16> -> tensor<1x64x1x18xf16>
    // CHECK:       %[[ADD0:.*]] = IE.Add(%[[CONV0]], %[[CONV1]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x64x1x18xf16>, tensor<1x64x1x18xf16> -> tensor<1x64x1x18xf16>

    // CHECK:       %[[SLICED_INPUT2:.*]] = IE.Slice %arg0 [0, 0, 18, 0] [1, 64, 1, 20] : tensor<1x64x20x20xf16> to tensor<1x64x1x20xf16>
    // CHECK:       %[[CONV2:.*]] = IE.Convolution(%[[SLICED_INPUT2]], %[[SLICED_FILTER2]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x64x1x20xf16>, tensor<64x64x1x3xf16> -> tensor<1x64x1x18xf16>

    // CHECK:       %[[ADD1:.*]] = IE.Add(%[[ADD0]], %[[CONV2]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x64x1x18xf16>, tensor<1x64x1x18xf16> -> tensor<1x64x1x18xf16>

    // CHECK:       %[[SLICED_INPUT3:.*]] = IE.Slice %arg0 [0, 0, 1, 0] [1, 64, 1, 20] : tensor<1x64x20x20xf16> to tensor<1x64x1x20xf16>
    // CHECK:       %[[CONV3:.*]] = IE.Convolution(%[[SLICED_INPUT3]], %[[SLICED_FILTER0]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x64x1x20xf16>, tensor<64x64x1x3xf16> -> tensor<1x64x1x18xf16>

    // CHECK:       %[[SLICED_INPUT4:.*]] = IE.Slice %arg0 [0, 0, 10, 0] [1, 64, 1, 20] : tensor<1x64x20x20xf16> to tensor<1x64x1x20xf16>
    // CHECK:       %[[CONV4:.*]] = IE.Convolution(%[[SLICED_INPUT4]], %[[SLICED_FILTER1]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x64x1x20xf16>, tensor<64x64x1x3xf16> -> tensor<1x64x1x18xf16>
    // CHECK:       %[[ADD2:.*]] = IE.Add(%[[CONV3]], %[[CONV4]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x64x1x18xf16>, tensor<1x64x1x18xf16> -> tensor<1x64x1x18xf16>

    // CHECK:       %[[SLICED_INPUT5:.*]] = IE.Slice %arg0 [0, 0, 19, 0] [1, 64, 1, 20] : tensor<1x64x20x20xf16> to tensor<1x64x1x20xf16>
    // CHECK:       %[[CONV5:.*]] = IE.Convolution(%[[SLICED_INPUT5]], %[[SLICED_FILTER2]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x64x1x20xf16>, tensor<64x64x1x3xf16> -> tensor<1x64x1x18xf16>

    // CHECK:       %[[ADD3:.*]] = IE.Add(%[[ADD2]], %[[CONV5]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x64x1x18xf16>, tensor<1x64x1x18xf16> -> tensor<1x64x1x18xf16>

    // CHECK:       %[[CONCAT:.*]]= IE.Concat(%[[ADD1]], %[[ADD3]]) {per_axis = {axis = 2 : i64}} : tensor<1x64x1x18xf16>, tensor<1x64x1x18xf16> -> tensor<1x64x2x18xf16>
    // CHECK:       return %[[CONCAT]]
}
