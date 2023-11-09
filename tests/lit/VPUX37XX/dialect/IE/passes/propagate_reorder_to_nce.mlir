//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --propagate-reorder-to-nce %s | FileCheck %s

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @PropagateReorderToConv
func.func @PropagateReorderToConv(%arg0: tensor<1x3x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16> {
    %cst = const.Declare tensor<3x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.Reorder<#NHWC>]    
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %1 = IE.Tanh(%0) : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>
    return %2 : tensor<1x3x512x512xf16>

    //CHECK:      [[CONV:%.*]] = IE.Convolution 
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:      [[REORDER:%.*]] = IE.Reorder([[CONV]]) {dstOrder = #NCHW}
    //CHECK:      [[TANH:%.*]] = IE.Tanh([[REORDER]]) 
    //CHECK-SAME: tensor<1x3x512x512xf16> -> tensor<1x3x512x512xf16>
    //CHECK:      return [[TANH]] : tensor<1x3x512x512xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NoPropagateReorderWithoutNCE
func.func @NoPropagateReorderWithoutNCE(%arg0: tensor<1x3x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16> {
    %0 = IE.Tanh(%arg0) : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %1 = IE.Reorder(%0) {dstOrder = #NCHW} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>
    return %1 : tensor<1x3x512x512xf16>

    //CHECK:      [[TANH:%.*]] = IE.Tanh
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:      [[REORDER:%.*]] = IE.Reorder([[TANH]]) {dstOrder = #NCHW}
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>
    //CHECK:      return [[REORDER]] : tensor<1x3x512x512xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @NoPropagateReorderTwoBranches
func.func @NoPropagateReorderTwoBranches(%arg0: tensor<1x3x512x512xf16, {order = #NHWC}>) -> (tensor<1x3x512x512xf16>, tensor<1x3x512x512xf16, {order = #NHWC}>) {
    %cst = const.Declare tensor<3x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.Reorder<#NHWC>]    
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %1 = IE.Tanh(%0) : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #NCHW} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>
    %3 = IE.MaxPool(%0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    return %2, %3 : tensor<1x3x512x512xf16>, tensor<1x3x512x512xf16, {order = #NHWC}>

    //CHECK:      [[CONV:%.*]] = IE.Convolution 
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:      [[TANH:%.*]] = IE.Tanh([[CONV]]) 
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:      [[REORDER:%.*]] = IE.Reorder([[TANH]]) {dstOrder = #NCHW}
    //CHECK:      [[MAXPOOL:%.*]] = IE.MaxPool([[CONV]])
    //CHECK:      return [[REORDER]], [[MAXPOOL]] : tensor<1x3x512x512xf16>, tensor<1x3x512x512xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>

// CHECK-LABEL: @NoPropagateNoSupportedReorder
func.func @NoPropagateNoSupportedReorder(%arg0: tensor<1x3x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16, {order = #map}> {
    %cst = const.Declare tensor<3x3x3x3xf16, {order = #NHWC}> = dense<1.0> : tensor<3x3x3x3xf16>, [#const.Reorder<#NHWC>]    
    %0 = IE.Convolution(%arg0, %cst) {dilations = [1, 1], pads_begin = [1, 1], pads_end = [1, 1], strides = [1, 1]} : tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %1 = IE.MVN(%0) {across_channels = false, eps = 1.0013580322265625E-5 : f64, normalize_variance = true} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    %2 = IE.Reorder(%1) {dstOrder = #map} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #map}>
    return %2 : tensor<1x3x512x512xf16, {order = #map}>

    //CHECK:      [[CONV:%.*]] = IE.Convolution 
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}>, tensor<3x3x3x3xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:      [[MVN:%.*]] = IE.MVN([[CONV]]) 
    //CHECK-SAME: tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16, {order = #NHWC}>
    //CHECK:      [[REORDER:%.*]] = IE.Reorder([[MVN]]) {dstOrder = #map}
    //CHECK:      return [[REORDER]] : tensor<1x3x512x512xf16, {order = #map}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SkipReorderWithBlockArg
func.func @SkipReorderWithBlockArg(%arg0: tensor<1x3x512x512xf16, {order = #NHWC}>) -> tensor<1x3x512x512xf16> {
    %0 = IE.Reorder(%arg0) {dstOrder = #NCHW} : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>
    return %0 : tensor<1x3x512x512xf16>

    // CHECK:   [[REORDER:%.*]] = IE.Reorder(%arg0) {
    // CHECK-SAME:      dstOrder = #NCHW
    // CHECK-SAME:  } : tensor<1x3x512x512xf16, {order = #NHWC}> -> tensor<1x3x512x512xf16>

    // CHECK:   return %0 : tensor<1x3x512x512xf16>
}
