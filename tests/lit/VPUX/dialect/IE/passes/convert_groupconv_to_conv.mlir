//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-groupconv-to-conv %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertGroupConvToConv
func.func @ConvertGroupConvToConv(%arg0: tensor<1x64x80x80xf16>) -> tensor<1x64x80x80xf16> {
    %weights = const.Declare tensor<64x16x3x3xf16> = dense<1.0> : tensor<64x16x3x3xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %weights, %bias) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x64x80x80xf16>, tensor<64x16x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>

    return %result : tensor<1x64x80x80xf16>

    // CHECK-NOT:   IE.GroupConvolution
    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<64x16x3x3xf16> = dense<1.000000e+00> : tensor<64x16x3x3xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>
    // CHECK:       [[VAL0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS0:%.*]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x16x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[BIAS0:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV_0:%.*]] = IE.Convolution([[VAL0]], [[WEIGHTS0]], [[BIAS0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[VAL1:%.*]] = IE.Slice %arg0 [0, 16, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS1:%.*]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x16x3x3xf16>, [#const.SubView<[16, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[BIAS1:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 16, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV_1:%.*]] = IE.Convolution([[VAL1]], [[WEIGHTS1]], [[BIAS1]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[VAL2:%.*]] = IE.Slice %arg0 [0, 32, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS2:%.*]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x16x3x3xf16>, [#const.SubView<[32, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[BIAS2:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 32, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV_2:%.*]] = IE.Convolution([[VAL2]], [[WEIGHTS2]], [[BIAS2]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[VAL3:%.*]] = IE.Slice %arg0 [0, 48, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS3:%.*]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x16x3x3xf16>, [#const.SubView<[48, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[BIAS3:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 48, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV_3:%.*]] = IE.Convolution([[VAL3]], [[WEIGHTS3]], [[BIAS3]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>
    // CHECK:       [[RESULT:%.*]] = IE.Concat([[CONV_0]], [[CONV_1]], [[CONV_2]], [[CONV_3]]) {per_axis = {axis = 1 : i64}}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<1x16x80x80xf16>, tensor<1x16x80x80xf16>, tensor<1x16x80x80xf16>
    // CHECK-SAME:      -> tensor<1x64x80x80xf16>
    // CHECK:       return [[RESULT]]
}

// -----

// CHECK-LABEL: @DoNotConvertGroupConvToConvWhenChannelNotAligned
func.func @DoNotConvertGroupConvToConvWhenChannelNotAligned(%arg0: tensor<1x16x80x80xf16>) -> tensor<1x16x80x80xf16> {
    %weights = const.Declare tensor<16x4x3x3xf16> = dense<1.0> : tensor<16x4x3x3xf16>
    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.0> : tensor<1x16x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %weights, %bias) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x16x80x80xf16>, tensor<16x4x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    return %result : tensor<1x16x80x80xf16>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<16x4x3x3xf16> = dense<1.000000e+00> : tensor<16x4x3x3xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK:       [[CONV_0:%.*]] = IE.GroupConvolution(%arg0, [[WEIGHTS]], [[BIAS]])
    // CHECK-SAME:      {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x4x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>
}

// -----

// CHECK-LABEL: @ConvertQuantizedGroupConvToConv
func.func @ConvertQuantizedGroupConvToConv(%arg0: tensor<1x64x80x80xf16>) -> tensor<1x64x80x80xf16> {
    %weights = const.Declare tensor<64x16x3x3xf16> = dense<0.5> : tensor<64x16x3x3xf16>
    %weights_low = const.Declare tensor<64x1x1x1xf16> = dense<0.000000e+00> : tensor<64x1x1x1xf16>
    %weights_high = const.Declare tensor<64x1x1x1xf16> = dense<1.000000e+00> : tensor<64x1x1x1xf16>
    %fq_weights = IE.FakeQuantize(%weights, %weights_low, %weights_high, %weights_low, %weights_high) {
                    auto_broadcast = #IE.auto_broadcast_type<NUMPY>,
                    levels = 256 : i64
                } : tensor<64x16x3x3xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16> -> tensor<64x16x3x3xf16>
    %bias = const.Declare tensor<1x64x1x1xf16> = dense<1.0> : tensor<1x64x1x1xf16>
    %result = IE.GroupConvolution(%arg0, %fq_weights, %bias) {dilations = [1, 1], groups = 4 : i64, pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x64x80x80xf16>, tensor<64x16x3x3xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x80x80xf16>

    return %result : tensor<1x64x80x80xf16>

    // CHECK-DAG:   [[WEIGHTS:%.*]] = const.Declare tensor<64x16x3x3xf16> = dense<5.000000e-01> : tensor<64x16x3x3xf16>
    // CHECK-DAG:   [[WEIGHTS_LOW:%.*]] = const.Declare tensor<64x1x1x1xf16> = dense<0.000000e+00> : tensor<64x1x1x1xf16>
    // CHECK-DAG:   [[WEIGHTS_HIGH:%.*]] = const.Declare tensor<64x1x1x1xf16> = dense<1.000000e+00> : tensor<64x1x1x1xf16>
    // CHECK:       [[WEIGHTS_FQ:%.*]] = IE.FakeQuantize([[WEIGHTS]], [[WEIGHTS_LOW]], [[WEIGHTS_HIGH]], [[WEIGHTS_LOW]], [[WEIGHTS_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK-SAME:      : tensor<64x16x3x3xf16>, tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16>
    // CHECK-SAME:      , tensor<64x1x1x1xf16>, tensor<64x1x1x1xf16> -> tensor<64x16x3x3xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x64x1x1xf16> = dense<1.000000e+00> : tensor<1x64x1x1xf16>

    // CHECK-DAG:   [[VAL0:%.*]] = IE.Slice %arg0 [0, 0, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS0:%.*]] = const.Declare tensor<16x16x3x3xf16> = dense<5.000000e-01>
    // CHECK-SAME:      : tensor<64x16x3x3xf16>, [#const.SubView<[0, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[WEIGHTS0_INPUT_LOW:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<0.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS0_OUTPUT_LOW:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<0.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS0_INPUT_HIGH:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS0_OUTPUT_HIGH:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK:       [[WEIGHTS0_FQ:%.*]] = IE.FakeQuantize([[WEIGHTS0]], [[WEIGHTS0_INPUT_LOW]]
    // CHECK-SAME:      , [[WEIGHTS0_INPUT_HIGH]], [[WEIGHTS0_OUTPUT_LOW]], [[WEIGHTS0_OUTPUT_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK-SAME:      : tensor<16x16x3x3xf16>, tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16>
    // CHECK-SAME:      , tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16> -> tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[BIAS0:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 0, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV0:%.*]] = IE.Convolution([[VAL0]], [[WEIGHTS0_FQ]], [[BIAS0]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    // CHECK-DAG:   [[VAL1:%.*]] = IE.Slice %arg0 [0, 16, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS1:%.*]] = const.Declare tensor<16x16x3x3xf16> = dense<5.000000e-01>
    // CHECK-SAME:      : tensor<64x16x3x3xf16>, [#const.SubView<[16, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[WEIGHTS1_INPUT_LOW:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<0.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS1_OUTPUT_LOW:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<0.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS1_INPUT_HIGH:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS1_OUTPUT_HIGH:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK:       [[WEIGHTS1_FQ:%.*]] = IE.FakeQuantize([[WEIGHTS1]], [[WEIGHTS1_INPUT_LOW]]
    // CHECK-SAME:      , [[WEIGHTS1_INPUT_HIGH]], [[WEIGHTS1_OUTPUT_LOW]], [[WEIGHTS1_OUTPUT_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK-SAME:      : tensor<16x16x3x3xf16>, tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16>
    // CHECK-SAME:      , tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16> -> tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[BIAS1:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 16, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV1:%.*]] = IE.Convolution([[VAL1]], [[WEIGHTS1_FQ]], [[BIAS1]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    // CHECK-DAG:   [[VAL2:%.*]] = IE.Slice %arg0 [0, 32, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS2:%.*]] = const.Declare tensor<16x16x3x3xf16> = dense<5.000000e-01>
    // CHECK-SAME:      : tensor<64x16x3x3xf16>, [#const.SubView<[32, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[WEIGHTS2_INPUT_LOW:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<0.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[32, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS2_OUTPUT_LOW:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<0.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[32, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS2_INPUT_HIGH:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[32, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS2_OUTPUT_HIGH:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[32, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK:       [[WEIGHTS2_FQ:%.*]] = IE.FakeQuantize([[WEIGHTS2]], [[WEIGHTS2_INPUT_LOW]]
    // CHECK-SAME:      , [[WEIGHTS2_INPUT_HIGH]], [[WEIGHTS2_OUTPUT_LOW]], [[WEIGHTS2_OUTPUT_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK-SAME:      : tensor<16x16x3x3xf16>, tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16>
    // CHECK-SAME:      , tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16> -> tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[BIAS2:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 32, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV2:%.*]] = IE.Convolution([[VAL2]], [[WEIGHTS2_FQ]], [[BIAS2]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    // CHECK-DAG:   [[VAL3:%.*]] = IE.Slice %arg0 [0, 48, 0, 0] [1, 16, 80, 80]
    // CHECK-SAME:      : tensor<1x64x80x80xf16> to tensor<1x16x80x80xf16>
    // CHECK-DAG:   [[WEIGHTS3:%.*]] = const.Declare tensor<16x16x3x3xf16> = dense<5.000000e-01>
    // CHECK-SAME:      : tensor<64x16x3x3xf16>, [#const.SubView<[48, 0, 0, 0], [16, 16, 3, 3]>]
    // CHECK-DAG:   [[WEIGHTS3_INPUT_LOW:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<0.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[48, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS3_OUTPUT_LOW:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<0.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[48, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS3_INPUT_HIGH:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[48, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK-DAG:   [[WEIGHTS3_OUTPUT_HIGH:%.*]] = const.Declare tensor<16x1x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<64x1x1x1xf16>, [#const.SubView<[48, 0, 0, 0], [16, 1, 1, 1]>]
    // CHECK:       [[WEIGHTS3_FQ:%.*]] = IE.FakeQuantize([[WEIGHTS3]], [[WEIGHTS3_INPUT_LOW]]
    // CHECK-SAME:      , [[WEIGHTS3_INPUT_HIGH]], [[WEIGHTS3_OUTPUT_LOW]], [[WEIGHTS3_OUTPUT_HIGH]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64}
    // CHECK-SAME:      : tensor<16x16x3x3xf16>, tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16>
    // CHECK-SAME:      , tensor<16x1x1x1xf16>, tensor<16x1x1x1xf16> -> tensor<16x16x3x3xf16>
    // CHECK-DAG:   [[BIAS3:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1x64x1x1xf16>, [#const.SubView<[0, 48, 0, 0], [1, 16, 1, 1]>]
    // CHECK:       [[CONV3:%.*]] = IE.Convolution([[VAL3]], [[WEIGHTS3_FQ]], [[BIAS3]])
    // CHECK-SAME:      {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<16x16x3x3xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x80x80xf16>

    // CHECK:       [[RESULT:%.*]] = IE.Concat([[CONV0]], [[CONV1]], [[CONV2]], [[CONV3]]) {per_axis = {axis = 1 : i64}}
    // CHECK-SAME:      : tensor<1x16x80x80xf16>, tensor<1x16x80x80xf16>, tensor<1x16x80x80xf16>, tensor<1x16x80x80xf16>
    // CHECK-SAME:      -> tensor<1x64x80x80xf16>
    // CHECK:       return [[RESULT]]
}
