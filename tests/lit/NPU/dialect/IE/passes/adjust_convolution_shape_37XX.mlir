//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-convolution-shape %s | FileCheck %s
// REQUIRES: arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FoldStrideIntoKernel
func.func @FoldStrideIntoKernel(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2x128x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x8x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 2]} : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x2xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
  return %0 : tensor<1x2x128x64xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x16x1x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT:%.+]] = IE.ShapeCast {shape = [1, 16, 128, 64]}
  // CHECK-SAME:      inputs(%arg0 : tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x16x128x64xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT]], [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFoldStrideIntoKernelForDifferentKXGreaterStride
func.func @NotFoldStrideIntoKernelForDifferentKXGreaterStride(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2x128x64xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<2x8x1x4xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x4xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 2], strides = [1, 2]} : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x4xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
    return %0 : tensor<1x2x128x64xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x8x1x4xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution(%arg0, [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 2], strides = [1, 2]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFoldStrideIntoKernelForWmodStideNone0
func.func @NotFoldStrideIntoKernelForWmodStideNone0(%arg0: tensor<1x8x128x128xf16, {order = #NHWC}>) -> tensor<1x2x128x43xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<2x8x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x8x1x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]} : tensor<1x8x128x128xf16, {order = #NHWC}>, tensor<2x8x1x3xf16, {order = #NHWC}> -> tensor<1x2x128x43xf16, {order = #NHWC}>
    return %0 : tensor<1x2x128x43xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x8x1x3xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution(%arg0, [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFoldStrideIntoKernelWhenChannelAligned
func.func @NotFoldStrideIntoKernelWhenChannelAligned(%arg0: tensor<1x16x128x128xf16, {order = #NHWC}>) -> tensor<1x16x128x43xf16, {order = #NHWC}> {
    %cst = const.Declare tensor<16x16x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<16x16x1x3xf16>, [#const.Reorder<#NHWC>]
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]} : tensor<1x16x128x128xf16, {order = #NHWC}>, tensor<16x16x1x3xf16, {order = #NHWC}> -> tensor<1x16x128x43xf16, {order = #NHWC}>
    return %0 : tensor<1x16x128x43xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x16x1x3xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution(%arg0, [[CST_WEIGHTS]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 3]}
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShape
func.func @AdjustConvolutionShape(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>
  
  // CHECK:       [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 45, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 3, 0, 0], [0, 42, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 6, 0, 0], [0, 39, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 9, 0, 0], [0, 36, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_4:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 12, 0, 0], [0, 33, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_5:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 15, 0, 0], [0, 30, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_6:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 18, 0, 0], [0, 27, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_7:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 21, 0, 0], [0, 24, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_8:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 24, 0, 0], [0, 21, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_9:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 27, 0, 0], [0, 18, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_10:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 30, 0, 0], [0, 15, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_11:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 33, 0, 0], [0, 12, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_12:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 36, 0, 0], [0, 9, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_13:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 39, 0, 0], [0, 6, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_14:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 42, 0, 0], [0, 3, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_15:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 45, 0, 0], [0, 0, 0, 0]>]
  // CHECK:       [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]], [[CST_WEIGHTS_4]], [[CST_WEIGHTS_5]], [[CST_WEIGHTS_6]], [[CST_WEIGHTS_7]], [[CST_WEIGHTS_8]], [[CST_WEIGHTS_9]], [[CST_WEIGHTS_10]], [[CST_WEIGHTS_11]], [[CST_WEIGHTS_12]], [[CST_WEIGHTS_13]], [[CST_WEIGHTS_14]], [[CST_WEIGHTS_15]]) {
  // CHECK:         per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK:         tensor<48x48x1x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeNoneSplatBias
func.func @AdjustConvolutionShapeNoneSplatBias(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x3x1x1xf16, {order = #NHWC}> = dense<[1.0, 2.0, 3.0]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x1xf16, {order = #NHWC}>, tensor<1x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>
  
  // CHECK:       [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 45, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 3, 0, 0], [0, 42, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 6, 0, 0], [0, 39, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 9, 0, 0], [0, 36, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_4:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 12, 0, 0], [0, 33, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_5:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 15, 0, 0], [0, 30, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_6:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 18, 0, 0], [0, 27, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_7:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 21, 0, 0], [0, 24, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_8:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 24, 0, 0], [0, 21, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_9:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 27, 0, 0], [0, 18, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_10:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 30, 0, 0], [0, 15, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_11:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 33, 0, 0], [0, 12, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_12:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 36, 0, 0], [0, 9, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_13:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 39, 0, 0], [0, 6, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_14:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 42, 0, 0], [0, 3, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_15:%.+]] = const.Declare tensor<3x48x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x1xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 3, 1, 1]>, #const.PadWithZero<[0, 45, 0, 0], [0, 0, 0, 0]>]
  // CHECK:       [[BIAS_CST:%.+]] = const.Declare tensor<1x48x1x1xf16, {order = #NHWC}> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Reorder<#NHWC>, #const.Broadcast<1 : i64, 48 : i64>, #const.Reshape<[1, 48, 1, 1]>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]], [[CST_WEIGHTS_4]], [[CST_WEIGHTS_5]], [[CST_WEIGHTS_6]], [[CST_WEIGHTS_7]], [[CST_WEIGHTS_8]], [[CST_WEIGHTS_9]], [[CST_WEIGHTS_10]], [[CST_WEIGHTS_11]], [[CST_WEIGHTS_12]], [[CST_WEIGHTS_13]], [[CST_WEIGHTS_14]], [[CST_WEIGHTS_15]]) {
  // CHECK:         per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK:         tensor<48x48x1x1xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPadingRight
func.func @AdjustConvolutionShapeWithKXGreat1andPadingRight(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x2xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>
  
  // CHECK:       [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 90, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 3, 0, 0], [0, 87, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 6, 0, 0], [0, 84, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 9, 0, 0], [0, 81, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_4:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 12, 0, 0], [0, 78, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_5:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 15, 0, 0], [0, 75, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_6:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 18, 0, 0], [0, 72, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_7:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 21, 0, 0], [0, 69, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_8:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 24, 0, 0], [0, 66, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_9:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 27, 0, 0], [0, 63, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_10:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 30, 0, 0], [0, 60, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_11:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 33, 0, 0], [0, 57, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_12:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 36, 0, 0], [0, 54, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_13:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 39, 0, 0], [0, 51, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_14:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 42, 0, 0], [0, 48, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_15:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 45, 0, 0], [0, 45, 0, 0]>]
  // CHECK:       [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]], [[CST_WEIGHTS_4]], [[CST_WEIGHTS_5]], [[CST_WEIGHTS_6]], [[CST_WEIGHTS_7]], [[CST_WEIGHTS_8]], [[CST_WEIGHTS_9]], [[CST_WEIGHTS_10]], [[CST_WEIGHTS_11]], [[CST_WEIGHTS_12]], [[CST_WEIGHTS_13]], [[CST_WEIGHTS_14]], [[CST_WEIGHTS_15]]) {
  // CHECK:         per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK:         tensor<48x96x1x1xf16, {order = #NHWC}>
  // CHECK:       [[FILTER_CAST:%.+]] = IE.ShapeCast {shape = [48, 48, 1, 2]} inputs([[FILTER_CST]] : tensor<48x96x1x1xf16, {order = #NHWC}>) -> tensor<48x48x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CAST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPadingLeft
func.func @AdjustConvolutionShapeWithKXGreat1andPadingLeft(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x2xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>
  
  // CHECK:       [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 45, 0, 0], [0, 45, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 48, 0, 0], [0, 42, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 51, 0, 0], [0, 39, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 54, 0, 0], [0, 36, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_4:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 57, 0, 0], [0, 33, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_5:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 60, 0, 0], [0, 30, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_6:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 63, 0, 0], [0, 27, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_7:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 66, 0, 0], [0, 24, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_8:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 69, 0, 0], [0, 21, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_9:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 72, 0, 0], [0, 18, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_10:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 75, 0, 0], [0, 15, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_11:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 78, 0, 0], [0, 12, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_12:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 81, 0, 0], [0, 9, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_13:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 84, 0, 0], [0, 6, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_14:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 87, 0, 0], [0, 3, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_15:%.+]] = const.Declare tensor<3x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x2xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 6, 1, 1]>, #const.PadWithZero<[0, 90, 0, 0], [0, 0, 0, 0]>]
  // CHECK:       [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]], [[CST_WEIGHTS_4]], [[CST_WEIGHTS_5]], [[CST_WEIGHTS_6]], [[CST_WEIGHTS_7]], [[CST_WEIGHTS_8]], [[CST_WEIGHTS_9]], [[CST_WEIGHTS_10]], [[CST_WEIGHTS_11]], [[CST_WEIGHTS_12]], [[CST_WEIGHTS_13]], [[CST_WEIGHTS_14]], [[CST_WEIGHTS_15]]) {
  // CHECK:         per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK:         tensor<48x96x1x1xf16, {order = #NHWC}>
  // CHECK:       [[FILTER_CAST:%.+]] = IE.ShapeCast {shape = [48, 48, 1, 2]} inputs([[FILTER_CST]] : tensor<48x96x1x1xf16, {order = #NHWC}>) -> tensor<48x48x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CAST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPadingLeftRight
func.func @AdjustConvolutionShapeWithKXGreat1andPadingLeftRight(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x3xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>

  // CHECK:       [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 45, 0, 0], [0, 90, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 48, 0, 0], [0, 87, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 51, 0, 0], [0, 84, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 54, 0, 0], [0, 81, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_4:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 57, 0, 0], [0, 78, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_5:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 60, 0, 0], [0, 75, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_6:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 63, 0, 0], [0, 72, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_7:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 66, 0, 0], [0, 69, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_8:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 69, 0, 0], [0, 66, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_9:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 72, 0, 0], [0, 63, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_10:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 75, 0, 0], [0, 60, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_11:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 78, 0, 0], [0, 57, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_12:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 81, 0, 0], [0, 54, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_13:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 84, 0, 0], [0, 51, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_14:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 87, 0, 0], [0, 48, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_15:%.+]] = const.Declare tensor<3x144x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 90, 0, 0], [0, 45, 0, 0]>]
  // CHECK:       [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]], [[CST_WEIGHTS_4]], [[CST_WEIGHTS_5]], [[CST_WEIGHTS_6]], [[CST_WEIGHTS_7]], [[CST_WEIGHTS_8]], [[CST_WEIGHTS_9]], [[CST_WEIGHTS_10]], [[CST_WEIGHTS_11]], [[CST_WEIGHTS_12]], [[CST_WEIGHTS_13]], [[CST_WEIGHTS_14]], [[CST_WEIGHTS_15]]) {
  // CHECK:         per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK:         tensor<48x144x1x1xf16, {order = #NHWC}>
  // CHECK:       [[FILTER_CAST:%.+]] = IE.ShapeCast {shape = [48, 48, 1, 3]} inputs([[FILTER_CST]] : tensor<48x144x1x1xf16, {order = #NHWC}>) -> tensor<48x48x1x3xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 48, 1080, 120]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x48x1080x120xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CAST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 1920]} inputs([[CONV_RET]] : tensor<1x48x1080x120xf16, {order = #NHWC}>) -> tensor<1x3x1080x1920xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat1andPadingStride
func.func @AdjustConvolutionShapeWithKXGreat1andPadingStride(%arg0: tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x3x1080x960xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x3x1x3xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0e-01> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst, %bias) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 0], strides = [1, 2]} : tensor<1x3x1080x1920xf16, {order = #NHWC}>, tensor<3x3x1x3xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x3x1080x960xf16, {order = #NHWC}>
  return %0 : tensor<1x3x1080x960xf16, {order = #NHWC}>
  // CHECK:       [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 93, 0, 0], [0, 90, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 99, 0, 0], [0, 84, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 105, 0, 0], [0, 78, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 111, 0, 0], [0, 72, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_4:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 117, 0, 0], [0, 66, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_5:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 123, 0, 0], [0, 60, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_6:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 129, 0, 0], [0, 54, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_7:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 135, 0, 0], [0, 48, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_8:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 141, 0, 0], [0, 42, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_9:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 147, 0, 0], [0, 36, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_10:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 153, 0, 0], [0, 30, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_11:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 159, 0, 0], [0, 24, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_12:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 165, 0, 0], [0, 18, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_13:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 171, 0, 0], [0, 12, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_14:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 177, 0, 0], [0, 6, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_15:%.+]] = const.Declare tensor<3x192x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x3x1x3xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[3, 9, 1, 1]>, #const.PadWithZero<[0, 183, 0, 0], [0, 0, 0, 0]>]

  // CHECK:       [[BIAS_CST:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<9.997550e-02> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]], [[CST_WEIGHTS_4]], [[CST_WEIGHTS_5]], [[CST_WEIGHTS_6]], [[CST_WEIGHTS_7]], [[CST_WEIGHTS_8]], [[CST_WEIGHTS_9]], [[CST_WEIGHTS_10]], [[CST_WEIGHTS_11]], [[CST_WEIGHTS_12]], [[CST_WEIGHTS_13]], [[CST_WEIGHTS_14]], [[CST_WEIGHTS_15]]) {
  // CHECK:         per_axis = #IE.Concat<axis = 0 : i64>}
  // CHECK:         tensor<48x192x1x1xf16, {order = #NHWC}>
  // CHECK:       [[FILTER_CAST:%.+]] = IE.ShapeCast {shape = [48, 96, 1, 2]} inputs([[FILTER_CST]] : tensor<48x192x1x1xf16, {order = #NHWC}>) -> tensor<48x96x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 96, 1080, 60]} inputs(%arg0 : tensor<1x3x1080x1920xf16, {order = #NHWC}>) -> tensor<1x96x1080x60xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CAST]], [[BIAS_CST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 3, 1080, 960]} inputs([[CONV_RET]] : tensor<1x48x1080x60xf16, {order = #NHWC}>) -> tensor<1x3x1080x960xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotAdjustConvolutionShapeWhenTensorFitCMX
func.func @NotAdjustConvolutionShapeWhenTensorFitCMX(%arg0: tensor<1x12x2x8xf16, {order = #NHWC}>) -> tensor<1x2x2x8xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x12x1x2xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x12x1x2xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 1], strides = [1, 1]} : tensor<1x12x2x8xf16, {order = #NHWC}>, tensor<2x12x1x2xf16, {order = #NHWC}> -> tensor<1x2x2x8xf16, {order = #NHWC}>
  return %0 : tensor<1x2x2x8xf16, {order = #NHWC}>
  
  // CHECK:       [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<2x12x1x2xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution(%arg0, [[CST_WEIGHTS_0]])
  // CHECK:       return [[CONV_RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat3andPadingBeginStride
func.func @AdjustConvolutionShapeWithKXGreat3andPadingBeginStride(%arg0: tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<4x4x1x4xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 2], pads_end = [0, 0], strides = [1, 2]} : tensor<1x4x1080x1920xf16, {order = #NHWC}>, tensor<4x4x1x4xf16, {order = #NHWC}> -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  return %0 : tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK:       [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 24, 0, 0], [0, 24, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 32, 0, 0], [0, 16, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 40, 0, 0], [0, 8, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 48, 0, 0], [0, 0, 0, 0]>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]]) {per_axis = #IE.Concat<axis = 0 : i64>}

  // CHECK:       [[FILTER_CAST:%.+]] = IE.ShapeCast {shape = [16, 32, 1, 2]} inputs([[FILTER_CST]] : tensor<16x64x1x1xf16, {order = #NHWC}>) -> tensor<16x32x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 32, 1080, 240]} inputs(%arg0 : tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x32x1080x240xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CAST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 4, 1080, 960]} inputs([[CONV_RET]] : tensor<1x16x1080x240xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat3andPadingEndStride
func.func @AdjustConvolutionShapeWithKXGreat3andPadingEndStride(%arg0: tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<4x4x1x4xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 2], strides = [1, 2]} : tensor<1x4x1080x1920xf16, {order = #NHWC}>, tensor<4x4x1x4xf16, {order = #NHWC}> -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  return %0 : tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK:       [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 0, 0, 0], [0, 48, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 8, 0, 0], [0, 40, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 16, 0, 0], [0, 32, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<4x64x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 24, 0, 0], [0, 24, 0, 0]>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]]) {per_axis = #IE.Concat<axis = 0 : i64>}

  // CHECK:       [[FILTER_CAST:%.+]] = IE.ShapeCast {shape = [16, 32, 1, 2]} inputs([[FILTER_CST]] : tensor<16x64x1x1xf16, {order = #NHWC}>) -> tensor<16x32x1x2xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 32, 1080, 240]} inputs(%arg0 : tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x32x1080x240xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CAST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 4, 1080, 960]} inputs([[CONV_RET]] : tensor<1x16x1080x240xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @AdjustConvolutionShapeWithKXGreat3andPadingBeginEndStride
func.func @AdjustConvolutionShapeWithKXGreat3andPadingBeginEndStride(%arg0: tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<4x4x1x4xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 2]} : tensor<1x4x1080x1920xf16, {order = #NHWC}>, tensor<4x4x1x4xf16, {order = #NHWC}> -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  return %0 : tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK:       [[CST_WEIGHTS_0:%.+]] = const.Declare tensor<4x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 28, 0, 0], [0, 52, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_1:%.+]] = const.Declare tensor<4x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 36, 0, 0], [0, 44, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_2:%.+]] = const.Declare tensor<4x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 44, 0, 0], [0, 36, 0, 0]>]
  // CHECK:       [[CST_WEIGHTS_3:%.+]] = const.Declare tensor<4x96x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<4x4x1x4xf16>, [#const.Reorder<#NHWC>, #const.Reshape<[4, 16, 1, 1]>, #const.PadWithZero<[0, 52, 0, 0], [0, 28, 0, 0]>]
  // CHECK:       [[FILTER_CST:%.+]] = IE.Concat([[CST_WEIGHTS_0]], [[CST_WEIGHTS_1]], [[CST_WEIGHTS_2]], [[CST_WEIGHTS_3]]) {per_axis = #IE.Concat<axis = 0 : i64>}

  // CHECK:       [[FILTER_CAST:%.+]] = IE.ShapeCast {shape = [16, 32, 1, 3]} inputs([[FILTER_CST]] : tensor<16x96x1x1xf16, {order = #NHWC}>) -> tensor<16x32x1x3xf16, {order = #NHWC}>
  // CHECK:       [[INPUT_CAST:%.+]] = IE.ShapeCast {shape = [1, 32, 1080, 240]} inputs(%arg0 : tensor<1x4x1080x1920xf16, {order = #NHWC}>) -> tensor<1x32x1080x240xf16, {order = #NHWC}>
  // CHECK:       [[CONV_RET:%.+]] = IE.Convolution([[INPUT_CAST]], [[FILTER_CAST]])
  // CHECK:       [[RET_CAST:%.+]] = IE.ShapeCast {shape = [1, 4, 1080, 960]} inputs([[CONV_RET]] : tensor<1x16x1080x240xf16, {order = #NHWC}>) -> tensor<1x4x1080x960xf16, {order = #NHWC}>
  // CHECK:       return [[RET_CAST]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>

// CHECK: func.func @NotAdjustConvWithMixedPrecisionFloatOutputQuantInput([[INPUT_DATA:%.+]]: tensor<1x3x320x320x!qElemType, {order = #NHWC}>)
func.func @NotAdjustConvWithMixedPrecisionFloatOutputQuantInput(%arg0: tensor<1x3x320x320x!qElemType, {order = #NHWC}>) -> tensor<1x32x160x160xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<32x3x3x3x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<32x3x3x3xf16>, [#const.ConvertElemType<si8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]}
            : tensor<1x3x320x320x!qElemType, {order = #NHWC}>, tensor<32x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x160x160xf16, {order = #NHWC}>

  return %result : tensor<1x32x160x160xf16, {order = #NHWC}>

  //CHECK:       [[VAL0:%.*]] = const.Declare tensor<32x3x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.ConvertElemType<si8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]
  //CHECK:       [[VAL1:%.*]] = IE.Convolution([[INPUT_DATA]], [[VAL0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x320x320x!qElemType, {order = #NHWC}>, tensor<32x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x160x160xf16, {order = #NHWC}>
  //CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>

// CHECK: func.func @NotAdjustConvWithMixedPrecisionFloatInputQuantWeights([[INPUT_DATA:%.+]]: tensor<1x3x320x320xf16, {order = #NHWC}>)
func.func @NotAdjustConvWithMixedPrecisionFloatInputQuantWeights(%arg0: tensor<1x3x320x320xf16, {order = #NHWC}>) -> tensor<1x32x160x160xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<32x3x3x3x!qElemType, {order = #NHWC}> = dense<1.0> : tensor<32x3x3x3xf16>, [#const.ConvertElemType<si8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]}
            : tensor<1x3x320x320xf16, {order = #NHWC}>, tensor<32x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x160x160xf16, {order = #NHWC}>

  return %result : tensor<1x32x160x160xf16, {order = #NHWC}>

  //CHECK:       [[VAL0:%.*]] = const.Declare tensor<32x3x3x3x!qElemType, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.ConvertElemType<si8>, #const.QuantCast<!qElemType>, #const.Reorder<#NHWC>]
  //CHECK:       [[VAL1:%.*]] = IE.Convolution([[INPUT_DATA]], [[VAL0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x320x320xf16, {order = #NHWC}>, tensor<32x3x3x3x!qElemType, {order = #NHWC}> -> tensor<1x32x160x160xf16, {order = #NHWC}>
  //CHECK:       return [[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType = !quant.uniform<i8:f16, 1.1534313725490195>

// CHECK: func.func @NotAdjustConvWithMixedPrecisionFloatInputQuantOutput([[INPUT_DATA:%.+]]: tensor<1x3x320x320xf16, {order = #NHWC}>)
func.func @NotAdjustConvWithMixedPrecisionFloatInputQuantOutput(%arg0: tensor<1x3x320x320xf16, {order = #NHWC}>) -> tensor<1x32x160x160x!qElemType, {order = #NHWC}> {
  %weights = const.Declare tensor<32x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<32x3x3x3xf16>, [#const.Reorder<#NHWC>]
  %result = IE.Convolution(%arg0, %weights) {
              dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]}
            : tensor<1x3x320x320xf16, {order = #NHWC}>, tensor<32x3x3x3xf16, {order = #NHWC}> -> tensor<1x32x160x160x!qElemType, {order = #NHWC}>

  return %result : tensor<1x32x160x160x!qElemType, {order = #NHWC}>

  //CHECK:       [[VAL0:%.*]] = const.Declare tensor<32x3x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<32x3x3x3xf16>, [#const.Reorder<#NHWC>]
  //CHECK:       [[VAL1:%.*]] = IE.Convolution([[INPUT_DATA]], [[VAL0]]) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [0, 0], strides = [2, 2]} : tensor<1x3x320x320xf16, {order = #NHWC}>, tensor<32x3x3x3xf16, {order = #NHWC}> -> tensor<1x32x160x160x!qElemType, {order = #NHWC}>
  //CHECK:       return [[VAL1]]
}
