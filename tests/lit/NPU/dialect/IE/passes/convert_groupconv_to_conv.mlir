//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//


// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-groupconv-to-conv %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertGroupConvToSingleConv
func.func @ConvertGroupConvToSingleConv(%arg0: tensor<1x64x80x80xf16>) -> tensor<1x64x80x80xf16> {
    %weights = const.Declare tensor<64x16x3x3xf16> = dense<1.0> : tensor<64x16x3x3xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %weights, %bias) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x64x80x80xf16>, tensor<64x16x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>

    return %result : tensor<1x64x80x80xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[ORG_WEIGHTS:%.+]] = const.Declare tensor<64x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>

    // CHECK-DAG:   [[WEIGHTS0:%.+]] = const.Declare tensor<16x64x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[0, 0, 0, 0], [16, 16, 3, 3]>, #const.PadWithZero<[0, 0, 0, 0], [0, 48, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS1:%.+]] = const.Declare tensor<16x64x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[16, 0, 0, 0], [16, 16, 3, 3]>, #const.PadWithZero<[0, 16, 0, 0], [0, 32, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS2:%.+]] = const.Declare tensor<16x64x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[32, 0, 0, 0], [16, 16, 3, 3]>, #const.PadWithZero<[0, 32, 0, 0], [0, 16, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS3:%.+]] = const.Declare tensor<16x64x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[48, 0, 0, 0], [16, 16, 3, 3]>, #const.PadWithZero<[0, 48, 0, 0], [0, 0, 0, 0]>]

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[WEIGHTS0]], [[WEIGHTS1]], [[WEIGHTS2]], [[WEIGHTS3]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:                      tensor<16x64x3x3xf16>, tensor<16x64x3x3xf16>, tensor<16x64x3x3xf16>, tensor<16x64x3x3xf16> -> tensor<64x64x3x3xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[CONCAT]], [[BIAS]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x64x80x80xf16>, tensor<64x64x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>

    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @ConvertGroupConvToMultiConvDueToNonConstWeights
func.func @ConvertGroupConvToMultiConvDueToNonConstWeights(%arg0: tensor<1x64x80x80xf16>, %arg1: tensor<64x16x3x3xf16>) -> tensor<1x64x80x80xf16> {
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %arg1, %bias) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x64x80x80xf16>, tensor<64x16x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>

    return %result : tensor<1x64x80x80xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>
    // CHECK:       [[INPUT0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS0:%.*]] = IE.Slice %arg1 [0, 0, 0, 0] [16, 16, 3, 3]
    // CHECK-SAME:      : tensor<64x16x3x3xf16> to tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[BIAS0:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV_0:%.*]] = IE.Convolution([[INPUT0]], [[WEIGHTS0]], [[BIAS0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    // CHECK-DAG:   [[INPUT1:%.*]] = IE.Slice %arg0 [0, 16, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS1:%.*]] = IE.Slice %arg1 [16, 0, 0, 0] [16, 16, 3, 3]
    // CHECK-SAME:      : tensor<64x16x3x3xf16> to tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[BIAS1:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 16, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV_1:%.*]] = IE.Convolution([[INPUT1]], [[WEIGHTS1]], [[BIAS1]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    // CHECK-DAG:   [[INPUT2:%.*]] = IE.Slice %arg0 [0, 32, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS2:%.*]] = IE.Slice %arg1 [32, 0, 0, 0] [16, 16, 3, 3]
    // CHECK-SAME:      : tensor<64x16x3x3xf16> to tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[BIAS2:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 32, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV_2:%.*]] = IE.Convolution([[INPUT2]], [[WEIGHTS2]], [[BIAS2]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    // CHECK-DAG:   [[INPUT3:%.*]] = IE.Slice %arg0 [0, 48, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS3:%.*]] = IE.Slice %arg1 [48, 0, 0, 0] [16, 16, 3, 3]
    // CHECK-SAME:      : tensor<64x16x3x3xf16> to tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[BIAS3:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 48, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV_3:%.*]] = IE.Convolution([[INPUT3]], [[WEIGHTS3]], [[BIAS3]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>
    // CHECK:       [[RESULT:%.*]] = IE.Concat([[CONV_0]], [[CONV_1]], [[CONV_2]], [[CONV_3]]) {per_axis = #IE.Concat<axis = 1 : i64>}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<1x16x80x80xf16>, tensor<1x16x80x80xf16>, tensor<1x16x80x80xf16>
    // CHECK-SAME:      -> tensor<1x64x80x80xf16>
    // CHECK:       return [[RESULT]]
}

// -----

// CHECK-LABEL: @ConvertGroupConvToSingleConvWithAsymmetricalWeights
func.func @ConvertGroupConvToSingleConvWithAsymmetricalWeights(%arg0: tensor<1x16x1x30xf16>) -> tensor<1x8x1x28xf16> {
    %weights = const.Declare tensor<8x8x1x3xf16> = dense<1.0> : tensor<2x4x8x3xf16>, [#const.Reshape<[8, 8, 3]>, #const.Reshape<[8, 8, 1, 3]>]
    %result = IE.GroupConvolution(%arg0, %weights) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x1x30xf16>, tensor<8x8x1x3xf16> -> tensor<1x8x1x28xf16>

    return %result : tensor<1x8x1x28xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[ORG_WEIGHTS:%.+]] = const.Declare tensor<8x8x1x3xf16> = dense<1.000000e+00> : tensor<2x4x8x3xf16>, [#const.Reshape<[8, 8, 3]>, #const.Reshape<[8, 8, 1, 3]>]

    // CHECK-DAG:   [[WEIGHTS0:%.+]] = const.Declare tensor<4x16x1x3xf16> = dense<1.000000e+00> : tensor<2x4x8x3xf16>
    // CHECK-SAME:                     [#const.Reshape<[8, 8, 3]>, #const.Reshape<[8, 8, 1, 3]>, #const.SubView<[0, 0, 0, 0], [4, 8, 1, 3]>, #const.PadWithZero<[0, 0, 0, 0], [0, 8, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS1:%.+]] = const.Declare tensor<4x16x1x3xf16> = dense<1.000000e+00> : tensor<2x4x8x3xf16>
    // CHECK-SAME:                     [#const.Reshape<[8, 8, 3]>, #const.Reshape<[8, 8, 1, 3]>, #const.SubView<[4, 0, 0, 0], [4, 8, 1, 3]>, #const.PadWithZero<[0, 8, 0, 0], [0, 0, 0, 0]>]

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[WEIGHTS0]], [[WEIGHTS1]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:                      tensor<4x16x1x3xf16>, tensor<4x16x1x3xf16> -> tensor<8x16x1x3xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[CONCAT]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x1x30xf16>, tensor<8x16x1x3xf16> -> tensor<1x8x1x28xf16>

    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @ConvertGroupConvToSingleConvWhenChannelNotAligned
func.func @ConvertGroupConvToSingleConvWhenChannelNotAligned(%arg0: tensor<1x16x80x80xf16>) -> tensor<1x16x80x80xf16> {
    %weights = const.Declare tensor<16x4x3x3xf16> = dense<1.0> : tensor<16x4x3x3xf16>
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %weights, %bias) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x80x80xf16>, tensor<16x4x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    return %result : tensor<1x16x80x80xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[ORG_WEIGHTS:%.+]] = const.Declare tensor<16x4x3x3xf16> = dense<1.000000e+00> : tensor<16x4x3x3xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>

    // CHECK-DAG:   [[WEIGHTS0:%.+]] = const.Declare tensor<4x16x3x3xf16> = dense<1.000000e+00> : tensor<16x4x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[0, 0, 0, 0], [4, 4, 3, 3]>, #const.PadWithZero<[0, 0, 0, 0], [0, 12, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS1:%.+]] = const.Declare tensor<4x16x3x3xf16> = dense<1.000000e+00> : tensor<16x4x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[4, 0, 0, 0], [4, 4, 3, 3]>, #const.PadWithZero<[0, 4, 0, 0], [0, 8, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS2:%.+]] = const.Declare tensor<4x16x3x3xf16> = dense<1.000000e+00> : tensor<16x4x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[8, 0, 0, 0], [4, 4, 3, 3]>, #const.PadWithZero<[0, 8, 0, 0], [0, 4, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS3:%.+]] = const.Declare tensor<4x16x3x3xf16> = dense<1.000000e+00> : tensor<16x4x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[12, 0, 0, 0], [4, 4, 3, 3]>, #const.PadWithZero<[0, 12, 0, 0], [0, 0, 0, 0]>]

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[WEIGHTS0]], [[WEIGHTS1]], [[WEIGHTS2]], [[WEIGHTS3]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:                      tensor<4x16x3x3xf16>, tensor<4x16x3x3xf16>, tensor<4x16x3x3xf16>, tensor<4x16x3x3xf16> -> tensor<16x16x3x3xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[CONCAT]], [[BIAS]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    // CHECK:       return [[CONV]]
}

// -----

// CHECK-LABEL: @NotConvertForDWConv
func.func @NotConvertForDWConv(%arg0: tensor<1x16x80x80xf16>) -> tensor<1x16x80x80xf16> {
    %weights = const.Declare tensor<16x1x3x3xf16> = dense<1.0> : tensor<16x1x3x3xf16>
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %weights, %bias) {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x80x80xf16>, tensor<16x1x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    return %result : tensor<1x16x80x80xf16>

    // CHECK-DAG:   [[WEIGHTS:%.+]] = const.Declare tensor<16x1x3x3xf16> = dense<1.000000e+00> : tensor<16x1x3x3xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK:       [[CONV_0:%.+]] = IE.GroupConvolution(%arg0, [[WEIGHTS]], [[BIAS]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 16 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x1x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>
}

// -----

// CHECK-LABEL: @ConvertQuantizedGroupConvToSingleConv
func.func @ConvertQuantizedGroupConvToSingleConv(%arg0: tensor<1x64x80x80xf16>) -> tensor<1x64x80x80xf16> {
    %weights = const.Declare tensor<64x16x3x3xf16> = dense<1.0> : tensor<64x16x3x3xf16>
    %weights_low = const.Declare tensor<64x1x1x1xf16> = dense<-1.270000e+02> : tensor<64x1x1x1xf16>
    %weights_high = const.Declare tensor<64x1x1x1xf16> = dense<1.270000e+02> : tensor<64x1x1x1xf16>
    %fq_weights = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                    levels = 255 : i64
                } : tensor<64x16x3x3xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16> -> tensor<64x16x3x3xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %fq_weights, %bias) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x64x80x80xf16>, tensor<64x16x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>

    return %result : tensor<1x64x80x80xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[ORG_WEIGHTS:%.+]] = const.Declare tensor<64x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-DAG:   [[FQ_LOW:%.+]] = const.Declare tensor<64x1x1x1xf16> = dense<-1.270000e+02> : tensor<64x1x1x1xf16>
    // CHECK-DAG:   [[FQ_HIGH:%.+]] = const.Declare tensor<64x1x1x1xf16> = dense<1.270000e+02> : tensor<64x1x1x1xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>

    // CHECK-DAG:   [[WEIGHTS0_SLICE:%.+]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[0, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[WEIGHTS0_PAD_AFTER:%.+]] = const.Declare tensor<16x48x3x3xf16> = dense<0.000000e+00> : tensor<16x48x3x3xf16>
    // CHECK-DAG:   [[WEIGHTS0:%.+]] = IE.Concat([[WEIGHTS0_SLICE]], [[WEIGHTS0_PAD_AFTER]]) {
    // CHECK-SAME:                     per_axis = #IE.Concat<axis = 1 : i64>} : tensor<16x16x3x3xf16>, tensor<16x48x3x3xf16> -> tensor<16x64x3x3xf16>

    // CHECK-DAG:   [[WEIGHTS1_PAD_BEFORE:%.+]] = const.Declare tensor<16x16x3x3xf16> = dense<0.000000e+00> : tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[WEIGHTS1_SLICE:%.+]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[16, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[WEIGHTS1_PAD_AFTER:%.+]] = const.Declare tensor<16x32x3x3xf16> = dense<0.000000e+00> : tensor<16x32x3x3xf16>
    // CHECK-DAG:   [[WEIGHTS1:%.+]] = IE.Concat([[WEIGHTS1_PAD_BEFORE]], [[WEIGHTS1_SLICE]], [[WEIGHTS1_PAD_AFTER]]) {
    // CHECK-SAME:                     per_axis = #IE.Concat<axis = 1 : i64>} : tensor<16x16x3x3xf16>, tensor<16x16x3x3xf16>, tensor<16x32x3x3xf16> -> tensor<16x64x3x3xf16>

    // CHECK-DAG:   [[WEIGHTS2_PAD_BEFORE:%.+]] = const.Declare tensor<16x32x3x3xf16> = dense<0.000000e+00> : tensor<16x32x3x3xf16>
    // CHECK-DAG:   [[WEIGHTS2_SLICE:%.+]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[32, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[WEIGHTS2_PAD_AFTER:%.+]] = const.Declare tensor<16x16x3x3xf16> = dense<0.000000e+00> : tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[WEIGHTS2:%.+]] = IE.Concat([[WEIGHTS2_PAD_BEFORE]], [[WEIGHTS2_SLICE]], [[WEIGHTS2_PAD_AFTER]]) {
    // CHECK-SAME:                     per_axis = #IE.Concat<axis = 1 : i64>} : tensor<16x32x3x3xf16>, tensor<16x16x3x3xf16>, tensor<16x16x3x3xf16> -> tensor<16x64x3x3xf16>

    // CHECK-DAG:   [[WEIGHTS3_PAD_BEFORE:%.+]] = const.Declare tensor<16x48x3x3xf16> = dense<0.000000e+00> : tensor<16x48x3x3xf16>
    // CHECK-DAG:   [[WEIGHTS3_SLICE:%.+]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-SAME:                     [#const.SubView<[48, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[WEIGHTS3:%.+]] = IE.Concat([[WEIGHTS3_PAD_BEFORE]], [[WEIGHTS3_SLICE]]) {
    // CHECK-SAME:                     per_axis = #IE.Concat<axis = 1 : i64>} : tensor<16x48x3x3xf16>, tensor<16x16x3x3xf16> -> tensor<16x64x3x3xf16>

    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[WEIGHTS0]], [[WEIGHTS1]], [[WEIGHTS2]], [[WEIGHTS3]]) {per_axis = #IE.Concat<axis = 0 : i64>}
    // CHECK-SAME:                      tensor<16x64x3x3xf16>, tensor<16x64x3x3xf16>, tensor<16x64x3x3xf16>, tensor<16x64x3x3xf16> -> tensor<64x64x3x3xf16>

    // CHECK:       [[FQ:%.+]] = IE.FakeQuantize([[CONCAT]], [[FQ_LOW]], [[FQ_HIGH]], [[FQ_LOW]], [[FQ_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<64x64x3x3xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16> -> tensor<64x64x3x3xf16>

    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[FQ]], [[BIAS]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x64x80x80xf16>, tensor<64x64x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>

    // CHECK:       return [[CONV]]
}


// -----

// CHECK-LABEL: @ConvertGroupConvToSingleConvOutChannelEqualGroup
func.func @ConvertGroupConvToSingleConvOutChannelEqualGroup(%arg0: tensor<1x64x80x80xf16>) -> tensor<1x2x80x80xf16> {
    %weights = const.Declare tensor<2x32x3x3xf16> = dense<1.0> : tensor<2x32x3x3xf16>
    %bias = const.Declare tensor<1x2x1x1xf16> = dense<1.0> : tensor<1x2x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %weights, %bias) {dilations = [1, 1], groups = 2 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x64x80x80xf16>, tensor<2x32x3x3xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x80x80xf16>

    return %result : tensor<1x2x80x80xf16>


    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[ORG_WEIGHTS:%.+]] = const.Declare tensor<2x32x3x3xf16> = dense<1.000000e+00> : tensor<2x32x3x3xf16>
    // CHECK-DAG:   [[BIAS:%.+]] = const.Declare tensor<1x2x1x1xf16> = dense<1.000000e+00> : tensor<1x2x1x1xf16>
    // CHECK-DAG:   [[WEIGHTS0:%.+]] = const.Declare tensor<1x64x3x3xf16> = dense<1.000000e+00> : tensor<2x32x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [1, 32, 3, 3]>, #const.PadWithZero<[0, 0, 0, 0], [0, 32, 0, 0]>]
    // CHECK-DAG:   [[WEIGHTS1:%.+]] = const.Declare tensor<1x64x3x3xf16> = dense<1.000000e+00> : tensor<2x32x3x3xf16>, [#const.SubView<[1, 0, 0, 0], [1, 32, 3, 3]>, #const.PadWithZero<[0, 32, 0, 0], [0, 0, 0, 0]>]
    // CHECK:       [[CONCAT:%.+]] = IE.Concat([[WEIGHTS0]], [[WEIGHTS1]]) {per_axis = #IE.Concat<axis = 0 : i64>} : tensor<1x64x3x3xf16>, tensor<1x64x3x3xf16> -> tensor<2x64x3x3xf16>
    // CHECK:       [[CONV:%.+]] = IE.Convolution(%arg0, [[CONCAT]], [[BIAS]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x64x80x80xf16>, tensor<2x64x3x3xf16>, tensor<1x2x1x1xf16> -> tensor<1x2x80x80xf16>
    // CHECK:       return [[CONV]]
}
