//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @FuseConvAndBias
func.func @FuseConvAndBias(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %filters = const.Declare tensor<16x3x3x3xf32> = dense<1.0> : tensor<16x3x3x3xf32>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x3x300x300xf32>, tensor<16x3x3x3xf32> -> tensor<1x16x300x300xf32>

    %bias = const.Declare tensor<1x16x1x1xf32> = dense<1.0> : tensor<1x16x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x16x300x300xf32>, tensor<1x16x1x1xf32> -> tensor<1x16x300x300xf32>

    return %1 : tensor<1x16x300x300xf32>

    // CHECK-DAG:   %[[FILTERS:.*]] = const.Declare tensor<16x3x3x3xf32> = dense<1.000000e+00> : tensor<16x3x3x3xf32>
    // CHECK-DAG:   %[[BIAS:.*]] = const.Declare tensor<1x16x1x1xf32> = dense<1.000000e+00> : tensor<1x16x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Convolution(%arg0, %[[FILTERS]], %[[BIAS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @GroupsToAttr
func.func @GroupsToAttr(%arg0: tensor<1x16x300x300xf32>) -> tensor<1x16x300x300xf32> {
    %filters = const.Declare tensor<16x1x1x3x3xf32> = dense<1.0> : tensor<16x1x1x3x3xf32>
    %0 = IE.GroupConvolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [1, 1],
            pads_end = [1, 1],
            dilations = [1, 1]
        } :
        tensor<1x16x300x300xf32>, tensor<16x1x1x3x3xf32> -> tensor<1x16x300x300xf32>

    return %0 : tensor<1x16x300x300xf32>

    // CHECK-DAG:       %[[FILTERS:.*]] = const.Declare tensor<16x1x3x3xf32> =
    // CHECK-SAM:       dense<1.000000e+00> : tensor<16x1x1x3x3xf32>, [#const.Reshape<[16, 1, 3, 3]>]
    // CHECK:       %[[VAL0:.*]] = IE.GroupConvolution(%arg0, %[[FILTERS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      groups = 16
    // CHECK-SAME:      pads_begin = [1, 1]
    // CHECK-SAME:      pads_end = [1, 1]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @NotFuseConvAndBias
func.func @NotFuseConvAndBias(%arg0: tensor<1x64x64x157xf32>) -> tensor<1x64x64x157xf32> {
    %filters = const.Declare tensor<64x64x1x3xf32> = dense<1.0> : tensor<64x64x1x3xf32>
    %0 = IE.Convolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 2],
            pads_end = [0, 0],
            dilations = [1, 1]
        } :
        tensor<1x64x64x157xf32>, tensor<64x64x1x3xf32> -> tensor<1x64x64x157xf32>

    %bias = const.Declare tensor<1x64x1x1xf32> = dense<1.0> : tensor<1x64x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x64x64x157xf32>, tensor<1x64x1x1xf32> -> tensor<1x64x64x157xf32>

    return %1 : tensor<1x64x64x157xf32>

    // CHECK-DAG:   %[[FILTERS:.*]] = const.Declare tensor<64x64x1x3xf32> = dense<1.000000e+00> : tensor<64x64x1x3xf32>
    // CHECK-DAG:   %[[BIAS:.*]] = const.Declare tensor<1x64x1x1xf32> = dense<1.000000e+00> : tensor<1x64x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Convolution(%arg0, %[[FILTERS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 2]
    // CHECK-SAME:      pads_end = [0, 0]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       %[[VAL1:.*]] = IE.ScaleShift(%[[VAL0]], %[[BIAS]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x64x64x157xf32>, tensor<1x64x1x1xf32> -> tensor<1x64x64x157xf32>
    // CHECK:       return %[[VAL1]]
}

// -----

// CHECK-LABEL: @NotFuseGroupConvAndBias
func.func @NotFuseGroupConvAndBias(%arg0: tensor<1x11x16x16xf32>) -> tensor<1x11x14x18xf32> {
    %filters = const.Declare tensor<11x1x1x3x3xf32> = dense<1.0> : tensor<11x1x1x3x3xf32>
    %0 = IE.GroupConvolution(%arg0, %filters)
        {
            strides = [1, 1],
            pads_begin = [0, 2],
            pads_end = [0, 2],
            dilations = [1, 1]
        } :
        tensor<1x11x16x16xf32>, tensor<11x1x1x3x3xf32> -> tensor<1x11x14x18xf32>

    %bias = const.Declare tensor<1x11x1x1xf32> = dense<1.0> : tensor<1x11x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x11x14x18xf32>, tensor<1x11x1x1xf32> -> tensor<1x11x14x18xf32>

    return %1 : tensor<1x11x14x18xf32>

    // CHECK-DAG:   %[[FILTERS:.*]] = const.Declare tensor<11x1x3x3xf32> =
    // CHECK-SAM:       dense<1.000000e+00> : tensor<11x1x1x3x3xf32>, [#const.Reshape<[11, 1, 3, 3]>]
    // CHECK-DAG:   %[[BIAS:.*]] = const.Declare tensor<1x11x1x1xf32> = dense<1.000000e+00> : tensor<1x11x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.GroupConvolution(%arg0, %[[FILTERS]])
    // CHECK-SAME:      dilations = [1, 1]
    // CHECK-SAME:      pads_begin = [0, 2]
    // CHECK-SAME:      pads_end = [0, 2]
    // CHECK-SAME:      strides = [1, 1]
    // CHECK:       %[[VAL1:.*]] = IE.ScaleShift(%[[VAL0]], %[[BIAS]]) {operandSegmentSizes = array<i32: 1, 0, 1>} : tensor<1x11x14x18xf32>, tensor<1x11x1x1xf32> -> tensor<1x11x14x18xf32>
    // CHECK:       return %[[VAL1]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @FuseSliceAndConv
func.func @FuseSliceAndConv(%arg0: tensor<1x2x128x64xf16, {order = #NHWC}>) -> tensor<1x2x128x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x1x1x1xf16, {order = #NHWC}> = dense<1.1> : tensor<2x1x1x1xf16, {order = #NHWC}>
  %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 64] : tensor<1x2x128x64xf16, {order = #NHWC}> to tensor<1x1x128x64xf16, {order = #NHWC}>
  %1 = IE.Convolution(%0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x1x128x64xf16, {order = #NHWC}>, tensor<2x1x1x1xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
  return %1 : tensor<1x2x128x64xf16, {order = #NHWC}>
  // CHECK:       [[FILTERS:%.*]] = const.Declare tensor<2x2x1x1xf16, {order = #NHWC}> = dense<1.099610e+00> : tensor<2x1x1x1xf16, {order = #NHWC}>,
  // CHECK-SAME:    #const.PadWithZero<[0, 0, 0, 0], [0, 1, 0, 0]>
  // CHECK:       [[RET:%.*]] = IE.Convolution(%arg0, [[FILTERS]]) {
  // CHECK-SAME:    dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK-SAME:    tensor<1x2x128x64xf16, {order = #NHWC}>, tensor<2x2x1x1xf16, {order = #NHWC}> -> tensor<1x2x128x64xf16, {order = #NHWC}>
  // CHECK:       return [[RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFuseSliceAndConvWhenSliceNotInDimC
func.func @NotFuseSliceAndConvWhenSliceNotInDimC(%arg0: tensor<1x1x128x128xf16, {order = #NHWC}>) -> tensor<1x2x64x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x1x2x1xf16, {order = #NHWC}> = dense<1.1> : tensor<2x1x2x1xf16, {order = #NHWC}>
  %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 64] : tensor<1x1x128x128xf16, {order = #NHWC}> to tensor<1x1x128x64xf16, {order = #NHWC}>
  %1 = IE.Convolution(%0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x1x128x64xf16, {order = #NHWC}>, tensor<2x1x2x1xf16, {order = #NHWC}> -> tensor<1x2x64x64xf16, {order = #NHWC}>
  return %1 : tensor<1x2x64x64xf16, {order = #NHWC}>
  // CHECK:       [[FILTERS:%.*]] = const.Declare tensor<2x1x2x1xf16, {order = #NHWC}> = dense<1.099610e+00> : tensor<2x1x2x1xf16, {order = #NHWC}>
  // CHECK:       [[SLICE_RET:%.*]] = IE.Slice %arg0
  // CHECK:       [[RET:%.*]] = IE.Convolution([[SLICE_RET]], [[FILTERS]])
  // CHECK:       return [[RET]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @NotFuseSliceAndConvWhenCostGreatThanExpand
func.func @NotFuseSliceAndConvWhenCostGreatThanExpand(%arg0: tensor<1x17x128x64xf16, {order = #NHWC}>) -> tensor<1x2x64x64xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x8x2x1xf16, {order = #NHWC}> = dense<1.1> : tensor<2x8x2x1xf16, {order = #NHWC}>
  %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 8, 128, 64] : tensor<1x17x128x64xf16, {order = #NHWC}> to tensor<1x8x128x64xf16, {order = #NHWC}>
  %1 = IE.Convolution(%0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x8x128x64xf16, {order = #NHWC}>, tensor<2x8x2x1xf16, {order = #NHWC}> -> tensor<1x2x64x64xf16, {order = #NHWC}>
  return %1 : tensor<1x2x64x64xf16, {order = #NHWC}>
  // CHECK:       [[FILTERS:%.*]] = const.Declare tensor<2x8x2x1xf16, {order = #NHWC}> = dense<1.099610e+00> : tensor<2x8x2x1xf16, {order = #NHWC}>
  // CHECK:       [[SLICE_RET:%.*]] = IE.Slice %arg0
  // CHECK:       [[RET:%.*]] = IE.Convolution([[SLICE_RET]], [[FILTERS]])
  // CHECK:       return [[RET]]
}

// -----

// CHECK-LABEL: @NotFuseSliceAndConvWhenNCHWLayout
func.func @NotFuseSliceAndConvWhenNCHWLayout(%arg0: tensor<1x2x128x64xf16>) -> tensor<1x2x64x64xf16> {
  %cst = const.Declare tensor<2x1x2x1xf16> = dense<1.1> : tensor<2x1x2x1xf16>
  %0 = IE.Slice %arg0 [0, 0, 0, 0] [1, 1, 128, 64] : tensor<1x2x128x64xf16> to tensor<1x1x128x64xf16>
  %1 = IE.Convolution(%0, %cst) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [2, 1]} : tensor<1x1x128x64xf16>, tensor<2x1x2x1xf16> -> tensor<1x2x64x64xf16>
  return %1 : tensor<1x2x64x64xf16>
  // CHECK:       [[FILTERS:%.*]] = const.Declare
  // CHECK:       [[SLICE:%.*]] = IE.Slice
  // CHECK:       [[RET:%.*]] = IE.Convolution
  // CHECK:       return [[RET]]
}

// -----

// CHECK-LABEL: @FuseTransposedConvAndBias
func.func @FuseTransposedConvAndBias(%arg0: tensor<1x3x64x64xf16>) -> tensor<1x16x129x129xf16> {
    %filters = const.Declare tensor<16x3x2x2xf16> = dense<1.000000e+00> : tensor<16x3x2x2xf16>
    %0 = IE.TransposedConvolution(%arg0, %filters)
        {
            dilations = [1, 1],
            operandSegmentSizes = array<i32: 1, 1, 0, 0>,
            output_padding = [1, 1],
            pads_begin = [0, 0],
            pads_end = [0, 0],
            strides = [2, 2]
        } :
        tensor<1x3x64x64xf16>, tensor<16x3x2x2xf16> -> tensor<1x16x129x129xf16>

    %bias = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    %1 = IE.ScaleShift(%0, %bias)
        {operandSegmentSizes = array<i32: 1, 0, 1>} :
        tensor<1x16x129x129xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x129x129xf16>

    return %1 : tensor<1x16x129x129xf16>

    // CHECK-DAG:   [[FILTERS:%.*]] = const.Declare tensor<16x3x2x2xf16> = dense<1.000000e+00> : tensor<16x3x2x2xf16>
    // CHECK-DAG:   [[BIAS:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK:       [[VAL0:%.*]] = IE.TransposedConvolution(%arg0, [[FILTERS]], [[BIAS]]) {
    // CHECK-SAME:      dilations = [1, 1],
    // CHECK-SAME:      operandSegmentSizes = array<i32: 1, 1, 0, 1>,
    // CHECK-SAME:      output_padding = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      strides = [2, 2]} : tensor<1x3x64x64xf16>, tensor<16x3x2x2xf16>, tensor<1x16x1x1xf16>
    // CHECK-SAME:      -> tensor<1x16x129x129xf16>

    // CHECK:       return [[VAL0]] : tensor<1x16x129x129xf16>
}
