//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --legalize-dilated-conv %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertDilatedConvolutionToConvolution1
func @ConvertDilatedConvolutionToConvolution1(%arg0: tensor<1x64x20x20xf16>) -> tensor<1x64x18x2xf16> {
    %FILTERS = const.Declare tensor<64x64x3x3xf16> = dense<1.000000e+00> : tensor<64x64x3x3xf16>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [1, 9], output_padding = [0, 0]} : tensor<1x64x20x20xf16>, tensor<64x64x3x3xf16> -> tensor<1x64x18x2xf16>
    return %RESULT : tensor<1x64x18x2xf16>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<64x64x3x3xf16> = dense<1.000000e+00> : tensor<64x64x3x3xf16>
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
    %FILTERS = const.Declare tensor<64x64x3x3xf16> = dense<1.000000e+00> : tensor<64x64x3x3xf16>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {strides = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], dilations = [9, 1], output_padding = [0, 0]} : tensor<1x64x20x20xf16>, tensor<64x64x3x3xf16> -> tensor<1x64x2x18xf16>
    return %RESULT : tensor<1x64x2x18xf16>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<64x64x3x3xf16> = dense<1.000000e+00> : tensor<64x64x3x3xf16>
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

// CHECK-LABEL: @ConvertXDilatedGroupConvolutionToGroupConvolution
func @ConvertXDilatedGroupConvolutionToGroupConvolution(%arg0: tensor<1x512x1x32xf16>) -> tensor<1x512x1x48xf16> {
    %FILTERS = const.Declare tensor<512x1x1x3xf16> = dense<1.000000e+00> : tensor<512x1x1x3xf16>
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [1, 8], groups = 512 : i64, pads_begin = [0, 16], pads_end = [0, 16], strides = [1, 1]} : tensor<1x512x1x32xf16>, tensor<512x1x1x3xf16> -> tensor<1x512x1x48xf16>
    return %RESULT : tensor<1x512x1x48xf16>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<512x1x1x3xf16> = dense<1.000000e+00> : tensor<512x1x1x3xf16>
    // CHECK:       %[[SLICED_FILTER0:.*]] = IE.Slice %[[CST]] [0, 0, 0, 0] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>
    // CHECK:       %[[SLICED_FILTER1:.*]] = IE.Slice %[[CST]] [0, 0, 0, 1] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>
    // CHECK:       %[[SLICED_FILTER2:.*]] = IE.Slice %[[CST]] [0, 0, 0, 2] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>

    // CHECK:       %[[LEFT_CST0:.*]] = const.Declare tensor<1x512x1x16xf16> = dense<0.000000e+00> : tensor<1x512x1x16xf16>
    // CHECK:       %[[CONCAT0:.*]] = IE.Concat(%[[LEFT_CST0]], %arg0) {per_axis = {axis = 3 : i64}} : tensor<1x512x1x16xf16>, tensor<1x512x1x32xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       %[[GROUPCONV0:.*]] = IE.GroupConvolution(%[[CONCAT0]], %[[SLICED_FILTER0]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<1x512x1x48xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x48xf16>

    // CHECK:       %[[LEFT_CST1:.*]] = const.Declare tensor<1x512x1x8xf16> = dense<0.000000e+00> : tensor<1x512x1x8xf16>
    // CHECK:       %[[RIGHT_CST1:.*]] = const.Declare tensor<1x512x1x8xf16> = dense<0.000000e+00> : tensor<1x512x1x8xf16>
    // CHECK:       %[[CONCAT1:.*]] = IE.Concat(%[[LEFT_CST1]], %arg0, %[[RIGHT_CST1]]) {per_axis = {axis = 3 : i64}} : tensor<1x512x1x8xf16>, tensor<1x512x1x32xf16>, tensor<1x512x1x8xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       %[[GROUPCONV1:.*]] = IE.GroupConvolution(%[[CONCAT1]], %[[SLICED_FILTER1]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<1x512x1x48xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       %[[ADD1:.*]] = IE.Add(%[[GROUPCONV0]], %[[GROUPCONV1]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x512x1x48xf16>, tensor<1x512x1x48xf16> -> tensor<1x512x1x48xf16>

    // CHECK:       %[[RIGHT_CST2:.*]] = const.Declare tensor<1x512x1x16xf16> = dense<0.000000e+00> : tensor<1x512x1x16xf16>
    // CHECK:       %[[CONCAT2:.*]] = IE.Concat(%arg0, %[[RIGHT_CST2]]) {per_axis = {axis = 3 : i64}} : tensor<1x512x1x32xf16>, tensor<1x512x1x16xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       %[[GROUPCONV2:.*]] = IE.GroupConvolution(%[[CONCAT2]], %[[SLICED_FILTER2]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<1x512x1x48xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x48xf16>
    // CHECK:       %[[ADD2:.*]] = IE.Add(%[[ADD1]], %[[GROUPCONV2]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x512x1x48xf16>, tensor<1x512x1x48xf16> -> tensor<1x512x1x48xf16>

    // CHECK:       return %[[ADD2]]
}

// CHECK-LABEL: @ConvertYDilatedGroupConvolutionToGroupConvolution
func @ConvertYDilatedGroupConvolutionToGroupConvolution(%arg0: tensor<1x512x32x1xf16>) -> tensor<1x512x48x1xf16> {
    %FILTERS = const.Declare tensor<512x1x3x1xf16> = dense<1.000000e+00> : tensor<512x1x3x1xf16>
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [8, 1], groups = 512 : i64, pads_begin = [16, 0], pads_end = [16, 0], strides = [1, 1]} : tensor<1x512x32x1xf16>, tensor<512x1x3x1xf16> -> tensor<1x512x48x1xf16>
    return %RESULT : tensor<1x512x48x1xf16>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<512x1x3x1xf16> = dense<1.000000e+00> : tensor<512x1x3x1xf16>
    // CHECK:       %[[SLICED_FILTER0:.*]] = IE.Slice %[[CST]] [0, 0, 0, 0] [512, 1, 1, 1] : tensor<512x1x3x1xf16> to tensor<512x1x1x1xf16>
    // CHECK:       %[[SLICED_FILTER1:.*]] = IE.Slice %[[CST]] [0, 0, 1, 0] [512, 1, 1, 1] : tensor<512x1x3x1xf16> to tensor<512x1x1x1xf16>
    // CHECK:       %[[SLICED_FILTER2:.*]] = IE.Slice %[[CST]] [0, 0, 2, 0] [512, 1, 1, 1] : tensor<512x1x3x1xf16> to tensor<512x1x1x1xf16>

    // CHECK:       %[[UP_CST0:.*]] = const.Declare tensor<1x512x16x1xf16> = dense<0.000000e+00> : tensor<1x512x16x1xf16>
    // CHECK:       %[[CONCAT0:.*]] = IE.Concat(%[[UP_CST0]], %arg0) {per_axis = {axis = 2 : i64}} : tensor<1x512x16x1xf16>, tensor<1x512x32x1xf16> -> tensor<1x512x48x1xf16>
    // CHECK:       %[[GROUPCONV0:.*]] = IE.GroupConvolution(%[[CONCAT0]], %[[SLICED_FILTER0]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<1x512x48x1xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x48x1xf16>

    // CHECK:       %[[UP_CST1:.*]] = const.Declare tensor<1x512x8x1xf16> = dense<0.000000e+00> : tensor<1x512x8x1xf16>
    // CHECK:       %[[BOTTOM_CST1:.*]] = const.Declare tensor<1x512x8x1xf16> = dense<0.000000e+00> : tensor<1x512x8x1xf16>
    // CHECK:       %[[CONCAT1:.*]] = IE.Concat(%[[UP_CST1]], %arg0, %[[BOTTOM_CST1]]) {per_axis = {axis = 2 : i64}} : tensor<1x512x8x1xf16>, tensor<1x512x32x1xf16>, tensor<1x512x8x1xf16> -> tensor<1x512x48x1xf16>
    // CHECK:       %[[GROUPCONV1:.*]] = IE.GroupConvolution(%[[CONCAT1]], %[[SLICED_FILTER1]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<1x512x48x1xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x48x1xf16>
    // CHECK:       %[[ADD1:.*]] = IE.Add(%[[GROUPCONV0]], %[[GROUPCONV1]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x512x48x1xf16>, tensor<1x512x48x1xf16> -> tensor<1x512x48x1xf16>

    // CHECK:       %[[BOTTOM_CST2:.*]] = const.Declare tensor<1x512x16x1xf16> = dense<0.000000e+00> : tensor<1x512x16x1xf16>
    // CHECK:       %[[CONCAT2:.*]] = IE.Concat(%arg0, %[[BOTTOM_CST2]]) {per_axis = {axis = 2 : i64}} : tensor<1x512x32x1xf16>, tensor<1x512x16x1xf16> -> tensor<1x512x48x1xf16>
    // CHECK:       %[[GROUPCONV2:.*]] = IE.GroupConvolution(%[[CONCAT2]], %[[SLICED_FILTER2]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      tensor<1x512x48x1xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x48x1xf16>
    // CHECK:       %[[ADD2:.*]] = IE.Add(%[[ADD1]], %[[GROUPCONV2]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x512x48x1xf16>, tensor<1x512x48x1xf16> -> tensor<1x512x48x1xf16>

    // CHECK:       return %[[ADD2]]
}

// CHECK-LABEL: @ConvertXDilatedStridedGroupConvolutionToGroupConvolution
func @ConvertXDilatedStridedGroupConvolutionToGroupConvolution(%arg0: tensor<1x512x1x32xf16>) -> tensor<1x512x1x24xf16> {
    %FILTERS = const.Declare tensor<512x1x1x3xf16> = dense<1.000000e+00> : tensor<512x1x1x3xf16>
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [1, 8], groups = 512 : i64, pads_begin = [0, 16], pads_end = [0, 16], strides = [1, 2]} : tensor<1x512x1x32xf16>, tensor<512x1x1x3xf16> -> tensor<1x512x1x24xf16>
    return %RESULT : tensor<1x512x1x24xf16>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<512x1x1x3xf16> = dense<1.000000e+00> : tensor<512x1x1x3xf16>
    // CHECK:       %[[SLICED_FILTER0:.*]] = IE.Slice %[[CST]] [0, 0, 0, 0] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>
    // CHECK:       %[[SLICED_FILTER1:.*]] = IE.Slice %[[CST]] [0, 0, 0, 1] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>
    // CHECK:       %[[SLICED_FILTER2:.*]] = IE.Slice %[[CST]] [0, 0, 0, 2] [512, 1, 1, 1] : tensor<512x1x1x3xf16> to tensor<512x1x1x1xf16>

    // CHECK:       %[[LEFT_CST0:.*]] = const.Declare tensor<1x512x1x16xf16> = dense<0.000000e+00> : tensor<1x512x1x16xf16>
    // CHECK:       %[[SLICED_INPUT0:.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 1, 31] : tensor<1x512x1x32xf16> to tensor<1x512x1x31xf16>
    // CHECK:       %[[CONCAT0:.*]] = IE.Concat(%[[LEFT_CST0]], %[[SLICED_INPUT0]]) {per_axis = {axis = 3 : i64}} : tensor<1x512x1x16xf16>, tensor<1x512x1x31xf16> -> tensor<1x512x1x47xf16>
    // CHECK:       %[[GROUPCONV0:.*]] = IE.GroupConvolution(%[[CONCAT0]], %[[SLICED_FILTER0]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
    // CHECK-SAME:      tensor<1x512x1x47xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x24xf16>

    // CHECK:       %[[LEFT_CST1:.*]] = const.Declare tensor<1x512x1x8xf16> = dense<0.000000e+00> : tensor<1x512x1x8xf16>
    // CHECK:       %[[RIGHT_CST1:.*]] = const.Declare tensor<1x512x1x7xf16> = dense<0.000000e+00> : tensor<1x512x1x7xf16>
    // CHECK:       %[[CONCAT1:.*]] = IE.Concat(%[[LEFT_CST1]], %arg0, %[[RIGHT_CST1]]) {per_axis = {axis = 3 : i64}} : tensor<1x512x1x8xf16>, tensor<1x512x1x32xf16>, tensor<1x512x1x7xf16> -> tensor<1x512x1x47xf16>
    // CHECK:       %[[GROUPCONV1:.*]] = IE.GroupConvolution(%[[CONCAT1]], %[[SLICED_FILTER1]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
    // CHECK-SAME:      tensor<1x512x1x47xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x24xf16>
    // CHECK:       %[[ADD1:.*]] = IE.Add(%[[GROUPCONV0]], %[[GROUPCONV1]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x512x1x24xf16>, tensor<1x512x1x24xf16> -> tensor<1x512x1x24xf16>

    // CHECK:       %[[RIGHT_CST2:.*]] = const.Declare tensor<1x512x1x15xf16> = dense<0.000000e+00> : tensor<1x512x1x15xf16>
    // CHECK:       %[[CONCAT2:.*]] = IE.Concat(%arg0, %[[RIGHT_CST2]]) {per_axis = {axis = 3 : i64}} : tensor<1x512x1x32xf16>, tensor<1x512x1x15xf16> -> tensor<1x512x1x47xf16>
    // CHECK:       %[[GROUPCONV2:.*]] = IE.GroupConvolution(%[[CONCAT2]], %[[SLICED_FILTER2]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]}
    // CHECK-SAME:      tensor<1x512x1x47xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x1x24xf16>
    // CHECK:       %[[ADD2:.*]] = IE.Add(%[[ADD1]], %[[GROUPCONV2]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x512x1x24xf16>, tensor<1x512x1x24xf16> -> tensor<1x512x1x24xf16>

    // CHECK:       return %[[ADD2]]
}

// CHECK-LABEL: @ConvertYDilatedStridedGroupConvolutionToGroupConvolution
func @ConvertYDilatedStridedGroupConvolutionToGroupConvolution(%arg0: tensor<1x512x32x1xf16>) -> tensor<1x512x24x1xf16> {
    %FILTERS = const.Declare tensor<512x1x3x1xf16> = dense<1.000000e+00> : tensor<512x1x3x1xf16>
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [8, 1], groups = 512 : i64, pads_begin = [16, 0], pads_end = [16, 0], strides = [2, 1]} : tensor<1x512x32x1xf16>, tensor<512x1x3x1xf16> -> tensor<1x512x24x1xf16>
    return %RESULT : tensor<1x512x24x1xf16>

    // CHECK:       %[[CST:.*]] = const.Declare tensor<512x1x3x1xf16> = dense<1.000000e+00> : tensor<512x1x3x1xf16>
    // CHECK:       %[[SLICED_FILTER0:.*]] = IE.Slice %[[CST]] [0, 0, 0, 0] [512, 1, 1, 1] : tensor<512x1x3x1xf16> to tensor<512x1x1x1xf16>
    // CHECK:       %[[SLICED_FILTER1:.*]] = IE.Slice %[[CST]] [0, 0, 1, 0] [512, 1, 1, 1] : tensor<512x1x3x1xf16> to tensor<512x1x1x1xf16>
    // CHECK:       %[[SLICED_FILTER2:.*]] = IE.Slice %[[CST]] [0, 0, 2, 0] [512, 1, 1, 1] : tensor<512x1x3x1xf16> to tensor<512x1x1x1xf16>

    // CHECK:       %[[UP_CST0:.*]] = const.Declare tensor<1x512x16x1xf16> = dense<0.000000e+00> : tensor<1x512x16x1xf16>
    // CHECK:       %[[SLICED_INPUT0:.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 512, 31, 1] : tensor<1x512x32x1xf16> to tensor<1x512x31x1xf16>
    // CHECK:       %[[CONCAT0:.*]] = IE.Concat(%[[UP_CST0]], %[[SLICED_INPUT0]]) {per_axis = {axis = 2 : i64}} : tensor<1x512x16x1xf16>, tensor<1x512x31x1xf16> -> tensor<1x512x47x1xf16>
    // CHECK:       %[[GROUPCONV0:.*]] = IE.GroupConvolution(%[[CONCAT0]], %[[SLICED_FILTER0]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]}
    // CHECK-SAME:      tensor<1x512x47x1xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x24x1xf16>

    // CHECK:       %[[UP_CST1:.*]] = const.Declare tensor<1x512x8x1xf16> = dense<0.000000e+00> : tensor<1x512x8x1xf16>
    // CHECK:       %[[BOTTOM_CST1:.*]] = const.Declare tensor<1x512x7x1xf16> = dense<0.000000e+00> : tensor<1x512x7x1xf16>
    // CHECK:       %[[CONCAT1:.*]] = IE.Concat(%[[UP_CST1]], %arg0, %[[BOTTOM_CST1]]) {per_axis = {axis = 2 : i64}} : tensor<1x512x8x1xf16>, tensor<1x512x32x1xf16>, tensor<1x512x7x1xf16> -> tensor<1x512x47x1xf16>
    // CHECK:       %[[GROUPCONV1:.*]] = IE.GroupConvolution(%[[CONCAT1]], %[[SLICED_FILTER1]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]}
    // CHECK-SAME:      tensor<1x512x47x1xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x24x1xf16>
    // CHECK:       %[[ADD1:.*]] = IE.Add(%[[GROUPCONV0]], %[[GROUPCONV1]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x512x24x1xf16>, tensor<1x512x24x1xf16> -> tensor<1x512x24x1xf16>

    // CHECK:       %[[BOTTOM_CST2:.*]] = const.Declare tensor<1x512x15x1xf16> = dense<0.000000e+00> : tensor<1x512x15x1xf16>
    // CHECK:       %[[CONCAT2:.*]] = IE.Concat(%arg0, %[[BOTTOM_CST2]]) {per_axis = {axis = 2 : i64}} : tensor<1x512x32x1xf16>, tensor<1x512x15x1xf16> -> tensor<1x512x47x1xf16>
    // CHECK:       %[[GROUPCONV2:.*]] = IE.GroupConvolution(%[[CONCAT2]], %[[SLICED_FILTER2]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]}
    // CHECK-SAME:      tensor<1x512x47x1xf16>, tensor<512x1x1x1xf16> -> tensor<1x512x24x1xf16>
    // CHECK:       %[[ADD2:.*]] = IE.Add(%[[ADD1]], %[[GROUPCONV2]]) {auto_broadcast = "NONE_OR_EXPLICIT"} : tensor<1x512x24x1xf16>, tensor<1x512x24x1xf16> -> tensor<1x512x24x1xf16>

    // CHECK:       return %[[ADD2]]
}
