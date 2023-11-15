//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-nce-ops-to-4d %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertNceOpsTo4DConvolution
func.func @ConvertNceOpsTo4DConvolution(%arg0: tensor<1x16x64xf16>) -> tensor<1x1x61xf16> {
    %FILTERS = const.Declare tensor<1x16x5xf16> = dense<1.000000e+00> : tensor<1x16x5xf16>
    %RESULT = IE.Convolution(%arg0, %FILTERS) {dilations = [2], pads_begin = [3], pads_end = [2], strides = [1]} : tensor<1x16x64xf16>, tensor<1x16x5xf16> -> tensor<1x1x61xf16>
    return %RESULT : tensor<1x1x61xf16>

    // CHECK:       %[[VAL0:.*]] = IE.Convolution
    // CHECK-SAME:      dilations = [1, 2]
    // CHECK-SAME:      pads_begin = [0, 3]
    // CHECK-SAME:      pads_end = [0, 2]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x16x1x64xf16>, tensor<1x16x1x5xf16> -> tensor<1x1x1x61xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 1, 61]} : tensor<1x1x1x61xf16> -> tensor<1x1x61xf16>
    // CHECK:       return %[[RESULT]]
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DConvolutionWithPostOp
func.func @ConvertNceOpsTo4DConvolutionWithPostOp(%arg0: tensor<1x16x64xf16>) -> tensor<1x1x64xf16> {
    %cts = const.Declare tensor<1x16x5xf16> = dense<1.000000e+00> : tensor<1x16x5xf16>
    %0 = IE.Convolution(%arg0, %cts) {dilations = [1], pads_begin = [2], pads_end = [2], strides = [1]} : tensor<1x16x64xf16>, tensor<1x16x5xf16> -> tensor<1x1x64xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x1x64xf16> -> tensor<1x1x64xf16>

    return %1 : tensor<1x1x64xf16>

    // CHECK:       [[VAL0:%.*]] = IE.Convolution
    // CHECK-SAME:       {dilations = [1, 1], pads_begin = [0, 2], pads_end = [0, 2],
    // CHECK-SAME:       strides = [1, 1]}
    // CHECK-SAME:       tensor<1x16x1x64xf16>, tensor<1x16x1x5xf16> -> tensor<1x1x1x64xf16>

    // CHECK:       [[VAL1:%.*]] = IE.Reshape([[VAL0]]) {shape_value = [1, 1, 64]} : tensor<1x1x1x64xf16> -> tensor<1x1x64xf16>

    // CHECK:       [[VAL2:%.*]] = IE.ReLU([[VAL1]]) : tensor<1x1x64xf16> -> tensor<1x1x64xf16>
    // CHECK:       return [[VAL2]] : tensor<1x1x64xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DGroupConvolution
func.func @ConvertNceOpsTo4DGroupConvolution(%arg0: tensor<1x16x30xf16>) -> tensor<1x8x28xf16>{
    %FILTERS = const.Declare tensor<8x8x3xf16> = dense<1.000000e+00> : tensor<2x4x8x3xf32>, [#const.Reshape<[8, 8, 3]>, #const.ConvertElemType<f16>]
    %RESULT = IE.GroupConvolution(%arg0, %FILTERS) {dilations = [1], groups = 2, pads_begin = [0], pads_end = [0], strides = [1]} : tensor<1x16x30xf16>, tensor<8x8x3xf16> -> tensor<1x8x28xf16>
    return %RESULT : tensor<1x8x28xf16>

    // CHECK:       %[[VAL0:.*]] = IE.GroupConvolution
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 2
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      tensor<1x16x1x30xf16>, tensor<8x8x1x3xf16> -> tensor<1x8x1x28xf16>
    // CHECK:       %[[RESULT:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 8, 28]} : tensor<1x8x1x28xf16> -> tensor<1x8x28xf16>
    // CHECK:       return %[[RESULT]]
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DDeconvolution
func.func @ConvertNceOpsTo4DDeconvolution(%arg0: tensor<1x384x1344xf16>) -> tensor<1x192x5380xf16> {
    %FILTERS = const.Declare tensor<384x192x8xf16> = dense<1.000000e+00> : tensor<384x192x8xf16>
    %RESULT = IE.Deconvolution(%arg0, %FILTERS) {
        dilations = [1], output_padding = [0], pads_begin = [0], pads_end = [0], strides = [4]} : tensor<1x384x1344xf16>, tensor<384x192x8xf16> -> tensor<1x192x5380xf16>

    return %RESULT : tensor<1x192x5380xf16>

    // CHECK: [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 384, 1, 1344]} : tensor<1x384x1344xf16> -> tensor<1x384x1x1344xf16>
    // CHECK: [[CST_0:%.+]] = const.Declare tensor<384x192x1x8xf16> = dense<1.000000e+00> : tensor<384x192x8xf16>, [#const.Reshape<[384, 192, 1, 8]>]

    // CHECK:       [[DECONV:%.*]] = IE.Deconvolution([[RESHAPE_INPUT]], [[CST_0]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      output_padding = [0, 0]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 4]
    // CHECK-SAME:      tensor<1x384x1x1344xf16>, tensor<384x192x1x8xf16> -> tensor<1x192x1x5380xf16>

    // CHECK: [[RESHAPE_OUTPUT:%.*]] = IE.Reshape([[DECONV]]) {shape_value = [1, 192, 5380]} : tensor<1x192x1x5380xf16> -> tensor<1x192x5380xf16>
    // CHECK: return [[RESHAPE_OUTPUT]] : tensor<1x192x5380xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DMaxpool
func.func @ConvertNceOpsTo4DMaxpool(%arg0: tensor<1x512x16xf16>) -> tensor<1x512x1xf16> {
    %RESULT = IE.MaxPool(%arg0) {
        kernel_size = [16], pads_begin = [0], pads_end = [0], rounding_type = #IE.rounding_type<FLOOR>, strides = [16]} : tensor<1x512x16xf16> -> tensor<1x512x1xf16>
    return %RESULT : tensor<1x512x1xf16>

    // CHECK: [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 512, 1, 16]} : tensor<1x512x16xf16> -> tensor<1x512x1x16xf16>

    // CHECK:       [[MAXPOOL:%.*]] = IE.MaxPool([[RESHAPE_INPUT]])
    // CHECK-SAME:      kernel_size = [1, 16]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:      strides = [1, 16]
    // CHECK-SAME:      tensor<1x512x1x16xf16> -> tensor<1x512x1x1xf16>

    // CHECK: [[RESHAPE_OUTPUT:%.*]] = IE.Reshape([[MAXPOOL]]) {shape_value = [1, 512, 1]} : tensor<1x512x1x1xf16> -> tensor<1x512x1xf16>
    // CHECK: return [[RESHAPE_OUTPUT]] : tensor<1x512x1xf16>
}

// -----

// CHECK-LABEL: @ConvertNceOpsTo4DAvgpool
func.func @ConvertNceOpsTo4DAvgpool(%arg0: tensor<1x512x16xf16>) -> tensor<1x512x1xf16> {
    %RESULT = IE.AvgPool(%arg0) {
        kernel_size = [16], pads_begin = [0], pads_end = [0], rounding_type = #IE.rounding_type<FLOOR>, strides = [16]} : tensor<1x512x16xf16> -> tensor<1x512x1xf16>
    return %RESULT : tensor<1x512x1xf16>

    // CHECK: [[RESHAPE_INPUT:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 512, 1, 16]} : tensor<1x512x16xf16> -> tensor<1x512x1x16xf16>

    // CHECK:       [[AVGPOOL:%.*]] = IE.AvgPool([[RESHAPE_INPUT]])
    // CHECK-SAME:      kernel_size = [1, 16]
    // CHECK-SAME:      pads_begin = [0, 0]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:      strides = [1, 16]
    // CHECK-SAME:      tensor<1x512x1x16xf16> -> tensor<1x512x1x1xf16>

    // CHECK: [[RESHAPE_OUTPUT:%.*]] = IE.Reshape([[AVGPOOL]]) {shape_value = [1, 512, 1]} : tensor<1x512x1x1xf16> -> tensor<1x512x1xf16>
    // CHECK: return [[RESHAPE_OUTPUT]] : tensor<1x512x1xf16>
}
