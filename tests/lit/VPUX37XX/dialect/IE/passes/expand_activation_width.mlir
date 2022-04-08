//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --expand-activation-width %s | FileCheck %s

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToConvolution
func @FusePermuteToConvolution(%arg0: tensor<1x16x16x23xf16, {order = #NHWC}>) -> tensor<1x16x15x21xf16> {
    %cst = const.Declare tensor<16x16x2x3xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x16x2x3xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    %CONV = IE.Convolution(%arg0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x16x23xf16, {order = #NHWC}>,
        tensor<16x16x2x3xf16, {order = #NHWC}> -> tensor<1x16x15x21xf16>

    return %CONV : tensor<1x16x15x21xf16>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<16x16x2x3xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<16x16x2x3xf32>,
    // CHECK-SAME:  [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:  pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:  pads_end = [0, 0, 0, 11]
    // CHECK-SAME:  } : tensor<1x16x16x23xf16, {order = #NHWC}> -> tensor<1x16x16x34xf16, {order = #NHWC}>

    // CHECK:   [[CONV:%.*]] = IE.Convolution([[EXPAND]], [[CST]]) {
    // CHECK-SAME:  dilations = [1, 1],
    // CHECK-SAME:  pads_begin = [0, 0],
    // CHECK-SAME:  pads_end = [0, 0],
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x16x34xf16, {order = #NHWC}>,
    // CHECK-SAME:    tensor<16x16x2x3xf16, {order = #NHWC}> -> tensor<1x16x15x32xf16>

    // CHECK:   [[SLICE:%.*]] = IE.Slice [[CONV]] [0, 0, 0, 0] [1, 16, 15, 21] :
    // CHECK-SAME:  tensor<1x16x15x32xf16> to tensor<1x16x15x21xf16>

    // CHECK:   return [[SLICE]] : tensor<1x16x15x21xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToMaxPool
func @FusePermuteToMaxPool(%arg0: tensor<1x16x16x23xf16, {order = #NHWC}>) -> tensor<1x16x14x21xf16> {
    %MAX_POOL = IE.MaxPool(%arg0) {
        kernel_size = [3, 3],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x16x16x23xf16, {order = #NHWC}> -> tensor<1x16x14x21xf16>

    return %MAX_POOL : tensor<1x16x14x21xf16>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:  pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:  pads_end = [0, 0, 0, 11]
    // CHECK-SAME:  } : tensor<1x16x16x23xf16, {order = #NHWC}> -> tensor<1x16x16x34xf16, {order = #NHWC}>

    // CHECK:   [[POOL:%.*]] = IE.MaxPool([[EXPAND]]) {
    // CHECK-SAME:  kernel_size = [3, 3],
    // CHECK-SAME:  pads_begin = [0, 0],
    // CHECK-SAME:  pads_end = [0, 0],
    // CHECK-SAME:  rounding_type = "FLOOR",
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x16x34xf16, {order = #NHWC}> -> tensor<1x16x14x32xf16>

    // CHECK:   [[SLICE:%.*]] = IE.Slice [[POOL]] [0, 0, 0, 0] [1, 16, 14, 21] :
    // CHECK-SAME:  tensor<1x16x14x32xf16> to tensor<1x16x14x21xf16>

    // CHECK:   return [[SLICE]] : tensor<1x16x14x21xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToGroupConv
func @FusePermuteToGroupConv(%arg0: tensor<1x16x16x23xf16, {order = #NHWC}>) -> tensor<1x16x15x21xf16> {
    %cst = const.Declare tensor<16x1x2x3xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<16x1x1x2x3xf32>,
        [#const.Reshape<[16, 1, 2, 3]>, #const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    %DWCONV = IE.GroupConvolution(%arg0, %cst) {
        dilations = [1, 1],
        groups = 16 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x16x23xf16, {order = #NHWC}>,
        tensor<16x1x2x3xf16, {order = #NHWC}> -> tensor<1x16x15x21xf16>

    return %DWCONV : tensor<1x16x15x21xf16>

    // CHECK:   [[CST:%.*]] = const.Declare tensor<16x1x2x3xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<16x1x1x2x3xf32>,
    // CHECK-SAME:  [#const.Reshape<[16, 1, 2, 3]>, #const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:  pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:  pads_end = [0, 0, 0, 11]
    // CHECK-SAME:  } : tensor<1x16x16x23xf16, {order = #NHWC}> -> tensor<1x16x16x34xf16, {order = #NHWC}>

    // CHECK:   [[DWCONV:%.*]] = IE.GroupConvolution([[EXPAND]], [[CST]]) {
    // CHECK-SAME:  dilations = [1, 1],
    // CHECK-SAME:  groups = 16 : i64,
    // CHECK-SAME:  pads_begin = [0, 0],
    // CHECK-SAME:  pads_end = [0, 0],
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x16x34xf16, {order = #NHWC}>,
    // CHECK-SAME:      tensor<16x1x2x3xf16, {order = #NHWC}> -> tensor<1x16x15x32xf16>

    // CHECK:   [[SLICE:%.*]] = IE.Slice [[DWCONV]] [0, 0, 0, 0] [1, 16, 15, 21] :
    // CHECK-SAME:  tensor<1x16x15x32xf16> to tensor<1x16x15x21xf16>

    // CHECK:   return [[SLICE]] : tensor<1x16x15x21xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToAvgPool
func @FusePermuteToAvgPool(%arg0: tensor<1x16x16x23xf16, {order = #NHWC}>) -> tensor<1x16x14x21xf16> {
    %AVG_POOL = IE.AvgPool(%arg0) {
        exclude_pads,
        kernel_size = [3, 3],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x16x16x23xf16, {order = #NHWC}> -> tensor<1x16x14x21xf16>

    return %AVG_POOL : tensor<1x16x14x21xf16>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:  pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:  pads_end = [0, 0, 0, 11]
    // CHECK-SAME:  } : tensor<1x16x16x23xf16, {order = #NHWC}> -> tensor<1x16x16x34xf16, {order = #NHWC}>

    // CHECK:   [[AVG_POOL:%.*]] = IE.AvgPool([[EXPAND]]) {
    // CHECK-SAME:  exclude_pads,
    // CHECK-SAME:  kernel_size = [3, 3],
    // CHECK-SAME:  pads_begin = [0, 0],
    // CHECK-SAME:  pads_end = [0, 0],
    // CHECK-SAME:  rounding_type = "FLOOR",
    // CHECK-SAME:  strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x16x16x34xf16, {order = #NHWC}> -> tensor<1x16x14x32xf16>

    // CHECK:   [[SLICE:%.*]] = IE.Slice [[AVG_POOL]] [0, 0, 0, 0] [1, 16, 14, 21] :
    // CHECK-SAME:  tensor<1x16x14x32xf16> to tensor<1x16x14x21xf16>

    // CHECK:   return [[SLICE]] : tensor<1x16x14x21xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToAdd
func @FusePermuteToAdd(%arg0: tensor<1x16x16x23xf16, {order = #NHWC}>) -> tensor<1x16x16x23xf16> {
    %cst = const.Declare tensor<1x16x16x23xf16, {order = #NHWC}> =
        dense<1.000000e+00> : tensor<1x16x16x23xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>]

    %ADD = IE.Add(%arg0, %cst) {
        auto_broadcast = "NUMPY"
    } : tensor<1x16x16x23xf16, {order = #NHWC}>,
        tensor<1x16x16x23xf16, {order = #NHWC}> -> tensor<1x16x16x23xf16>

    return %ADD : tensor<1x16x16x23xf16>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:  pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:  pads_end = [0, 0, 0, 9]
    // CHECK-SAME:  } : tensor<1x16x16x23xf16, {order = #NHWC}> -> tensor<1x16x16x32xf16, {order = #NHWC}>

    // CHECK:   [[EXPANDED_CST:%.*]] = const.Declare tensor<1x16x16x32xf16, {order = #NHWC}> =
    // CHECK-SAME:  dense<1.000000e+00> : tensor<1x16x16x23xf32>,
    // CHECK-SAME:  [
    // CHECK-SAME:      #const.ConvertElemType<f16>,
    // CHECK-SAME:      #const.Reorder<#NHWC>,
    // CHECK-SAME:      #const.PadWithZero<[0, 0, 0, 0], [0, 0, 0, 9]>
    // CHECK-SAME:  ]

    // CHECK:   [[ADD:%.*]] = IE.Add([[EXPAND]], [[EXPANDED_CST]]) {
    // CHECK-SAME:  auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x16x16x32xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x16x16x32xf16, {order = #NHWC}> -> tensor<1x16x16x32xf16>

    // CHECK:   [[SLICE:%.*]] = IE.Slice [[ADD]] [0, 0, 0, 0] [1, 16, 16, 23] :
    // CHECK-SAME:  tensor<1x16x16x32xf16> to tensor<1x16x16x23xf16>

    // CHECK:   return [[SLICE]] : tensor<1x16x16x23xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @FusePermuteToAddSameInput
func @FusePermuteToAddSameInput(%arg0: tensor<1x16x16x23xf16, {order = #NHWC}>) -> tensor<1x16x16x23xf16> {
    %ADD = IE.Add(%arg0, %arg0) {
        auto_broadcast = "NUMPY"
    } : tensor<1x16x16x23xf16, {order = #NHWC}>,
        tensor<1x16x16x23xf16, {order = #NHWC}> -> tensor<1x16x16x23xf16>

    return %ADD : tensor<1x16x16x23xf16>

    // CHECK:   [[EXPAND:%.*]] = IE.Expand(%arg0) {
    // CHECK-SAME:  pads_begin = [0, 0, 0, 0],
    // CHECK-SAME:  pads_end = [0, 0, 0, 9]
    // CHECK-SAME:  } : tensor<1x16x16x23xf16, {order = #NHWC}> -> tensor<1x16x16x32xf16, {order = #NHWC}>

    // CHECK:   [[ADD:%.*]] = IE.Add([[EXPAND]], [[EXPAND]]) {
    // CHECK-SAME:  auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x16x16x32xf16, {order = #NHWC}>,
    // CHECK-SAME:  tensor<1x16x16x32xf16, {order = #NHWC}> -> tensor<1x16x16x32xf16>

    // CHECK:   [[SLICE:%.*]] = IE.Slice [[ADD]] [0, 0, 0, 0] [1, 16, 16, 23] :
    // CHECK-SAME:  tensor<1x16x16x32xf16> to tensor<1x16x16x23xf16>

    // CHECK:   return [[SLICE]] : tensor<1x16x16x23xf16>
}
