//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --adjust-groupconv-shape %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @adjustGroupConvInput
func.func @adjustGroupConvInput(%arg0: tensor<1x2x64x512xf16, {order = #NHWC}>) -> tensor<1x2x64x512xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x2x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x2x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.GroupConvolution(%arg0, %cst, %bias) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2x64x512xf16, {order = #NHWC}>, tensor<2x1x1x1xf16, {order = #NHWC}>, tensor<1x2x1x1xf16, {order = #NHWC}> -> tensor<1x2x64x512xf16, {order = #NHWC}>
  return %0 : tensor<1x2x64x512xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}>
  // CHECK-DAG:   [[CST_BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}>
  // CHECK:   [[INPUT:%.+]] = IE.ShapeCast {shape = [1, 16, 64, 64]}
  // CHECK-SAME:     inputs(%arg0 : tensor<1x2x64x512xf16, {order = #NHWC}>) -> tensor<1x16x64x64xf16, {order = #NHWC}>
  // CHECK:   [[CONV_RET:%.+]] = IE.GroupConvolution([[INPUT]], [[CST_WEIGHTS]], [[CST_BIAS]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK:   [[RESULT:%.+]] = IE.ShapeCast {shape = [1, 2, 64, 512]}
  // CHECK-SAME:    inputs([[CONV_RET]] : tensor<1x16x64x64xf16, {order = #NHWC}>) -> tensor<1x2x64x512xf16, {order = #NHWC}>
  // CHECK:   return [[RESULT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @adjustGroupConvInputFailForCSTShapeNotAlign
func.func @adjustGroupConvInputFailForCSTShapeNotAlign(%arg0: tensor<1x2x64x512xf16, {order = #NHWC}>) -> tensor<1x2x64x512xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<2x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<2x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %0 = IE.GroupConvolution(%arg0, %cst, %bias) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x2x64x512xf16, {order = #NHWC}>, tensor<2x1x1x1xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x2x64x512xf16, {order = #NHWC}>
  return %0 : tensor<1x2x64x512xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<2x1x1x1xf16, {order = #NHWC}>
  // CHECK-DAG:   [[CST_BIAS:%.+]] = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}>
  // CHECK:       [[RESULT:%.+]] = IE.GroupConvolution(%arg0, [[CST_WEIGHTS]], [[CST_BIAS]]) {dilations = [1, 1], groups = 2 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK:       return [[RESULT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @adjustGroupConvInputForDimCFromDimHW
func.func @adjustGroupConvInputForDimCFromDimHW(%arg0: tensor<1x3x64x52xf16, {order = #NHWC}>) -> tensor<1x3x64x52xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x1x1x1xf16, {order = #NHWC}> = dense<[[[[1.0]]], [[[2.0]]], [[[3.0]]]]> : tensor<3x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x3x1x1xf16, {order = #NHWC}> = dense<[4.0, 5.0, 6.0]> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Reorder<#NHWC>]
  %0 = IE.GroupConvolution(%arg0, %cst, %bias) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x64x52xf16, {order = #NHWC}>, tensor<3x1x1x1xf16, {order = #NHWC}>, tensor<1x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x64x52xf16, {order = #NHWC}>
  return %0 : tensor<1x3x64x52xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<48x1x1x1xf16, {order = #NHWC}>
  // CHECK-SAME:        #const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 48 : i64>
  // CHECK-DAG:   [[CST_BIAS:%.+]] = const.Declare tensor<1x48x1x1xf16, {order = #NHWC}>
  // CHECK-SAME:        #const.Reorder<#NHWC>, #const.Broadcast<1 : i64, 48 : i64>
  // CHECK:   [[INPUT:%.+]] = IE.ShapeCast {shape = [1, 48, 16, 13]}
  // CHECK-SAME:    inputs(%arg0 : tensor<1x3x64x52xf16, {order = #NHWC}>) -> tensor<1x48x16x13xf16, {order = #NHWC}>
  // CHECK:   [[CONV_RET:%.+]] = IE.GroupConvolution([[INPUT]], [[CST_WEIGHTS]], [[CST_BIAS]]) {dilations = [1, 1], groups = 48 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK:   [[RESULT:%.+]] = IE.ShapeCast {shape = [1, 3, 64, 52]}
  // CHECK-SAME:    inputs(%1 : tensor<1x48x16x13xf16, {order = #NHWC}>) -> tensor<1x3x64x52xf16, {order = #NHWC}>
  // CHECK:   return [[RESULT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @adjustGroupConvInputForElem
func.func @adjustGroupConvInputForElem(%arg0: tensor<1x3x64x52xf16, {order = #NHWC}>) -> tensor<1x3x64x52xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<3x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<3x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x3x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<3xf16>, [#const.Reshape<[1, 3, 1, 1]>, #const.Reorder<#NHWC>]
  %0 = IE.GroupConvolution(%arg0, %cst, %bias) {dilations = [1, 1], groups = 3 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x3x64x52xf16, {order = #NHWC}>, tensor<3x1x1x1xf16, {order = #NHWC}>, tensor<1x3x1x1xf16, {order = #NHWC}> -> tensor<1x3x64x52xf16, {order = #NHWC}>
  return %0 : tensor<1x3x64x52xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}>
  // CHECK-DAG:   [[CST_BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}>
  // CHECK:   [[INPUT:%.+]] = IE.ShapeCast {shape = [1, 16, 26, 24]}
  // CHECK-SAME:    inputs(%arg0 : tensor<1x3x64x52xf16, {order = #NHWC}>) -> tensor<1x16x26x24xf16, {order = #NHWC}>
  // CHECK:   [[CONV_RET:%.+]] = IE.GroupConvolution([[INPUT]], [[CST_WEIGHTS]], [[CST_BIAS]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK:   [[RESULT:%.+]] = IE.ShapeCast {shape = [1, 3, 64, 52]}
  // CHECK-SAME:    inputs(%1 : tensor<1x16x26x24xf16, {order = #NHWC}>) -> tensor<1x3x64x52xf16, {order = #NHWC}>
  // CHECK:   return [[RESULT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @adjustGroupConvInputForElemAndReshapeCst
func.func @adjustGroupConvInputForElemAndReshapeCst(%arg0: tensor<1x17x64x52xf16, {order = #NHWC}>) -> tensor<1x17x64x52xf16, {order = #NHWC}> {
  %cst = const.Declare tensor<17x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01> : tensor<17x1x1x1xf16>, [#const.Reorder<#NHWC>]
  %bias = const.Declare tensor<1x17x1x1xf16, {order = #NHWC}> = dense<1.0> : tensor<17xf16>, [#const.Reshape<[1, 17, 1, 1]>, #const.Reorder<#NHWC>]
  %0 = IE.GroupConvolution(%arg0, %cst, %bias) {dilations = [1, 1], groups = 17 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x17x64x52xf16, {order = #NHWC}>, tensor<17x1x1x1xf16, {order = #NHWC}>, tensor<1x17x1x1xf16, {order = #NHWC}> -> tensor<1x17x64x52xf16, {order = #NHWC}>
  return %0 : tensor<1x17x64x52xf16, {order = #NHWC}>

  // CHECK-DAG:   [[CST_WEIGHTS:%.+]] = const.Declare tensor<16x1x1x1xf16, {order = #NHWC}> = dense<1.250000e-01>
  // CHECK-SAME:    tensor<17x1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 16 : i64>]
  // CHECK-DAG:   [[CST_BIAS:%.+]] = const.Declare tensor<1x16x1x1xf16, {order = #NHWC}> = dense<1.000000e+00>
  // CHECK-SAME:    tensor<17xf16>, [#const.Reshape<[1, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Broadcast<1 : i64, 16 : i64>]
  // CHECK:   [[INPUT:%.+]] = IE.ShapeCast {shape = [1, 16, 68, 52]}
  // CHECK-SAME:    inputs(%arg0 : tensor<1x17x64x52xf16, {order = #NHWC}>) -> tensor<1x16x68x52xf16, {order = #NHWC}>
  // CHECK:   [[CONV_RET:%.+]] = IE.GroupConvolution([[INPUT]], [[CST_WEIGHTS]], [[CST_BIAS]]) {dilations = [1, 1], groups = 16 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
  // CHECK:   [[RESULT:%.+]] = IE.ShapeCast {shape = [1, 17, 64, 52]}
  // CHECK-SAME:    inputs(%1 : tensor<1x16x68x52xf16, {order = #NHWC}>) -> tensor<1x17x64x52xf16, {order = #NHWC}>
  // CHECK:   return [[RESULT]]
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL: @adjustGroupConvInputNotAlignForElemAndReshapeCst
func.func @adjustGroupConvInputNotAlignForElemAndReshapeCst(%arg0 : tensor<1x1x289x289xf16, {order = #NHWC}>) -> (tensor<1x1x289x289xf16, {order = #NHWC}>) {
    %filter = const.Declare tensor<1x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reorder<#NHWC>]
    %group_conv = IE.GroupConvolution(%arg0, %filter) {
        dilations = [1, 1],
        groups = 1 : i64,
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
        } : tensor<1x1x289x289xf16, {order = #NHWC}>, tensor<1x1x1x1xf16, {order = #NHWC}> -> tensor<1x1x289x289xf16, {order = #NHWC}>

    return %group_conv : tensor<1x1x289x289xf16, {order = #NHWC}>
    // CHECK:       const.Declare
    // CHECK-SAME{{LITERAL}}:  tensor<289x1x1x1xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<1x1x1x1xf16>, [#const.Reshape<[1, 1, 1, 1]>, #const.Reorder<#NHWC>, #const.Broadcast<0 : i64, 289 : i64>]
    // CHECK:       IE.ShapeCast
    // CHECK-SAME{{LITERAL}}:  {shape = [1, 289, 17, 17]} inputs(%arg0 : tensor<1x1x289x289xf16, {order = #NHWC}>) -> tensor<1x289x17x17xf16, {order = #NHWC}>
    // CHECK:       IE.GroupConvolution
    // CHECK-SAME{{LITERAL}}:  {dilations = [1, 1], groups = 289 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]}
    // CHECK-SAME   tensor<1x289x17x17xf16, {order = #NHWC}>, tensor<289x1x1x1xf16, {order = #NHWC}> -> tensor<1x289x17x17xf16, {order = #NHWC}>
    // CHECK:       IE.ShapeCast
    // CHECK-SAME{{LITERAL}}:  {shape = [1, 1, 289, 289]} inputs(%1 : tensor<1x289x17x17xf16, {order = #NHWC}>) -> tensor<1x1x289x289xf16, {order = #NHWC}>
}
