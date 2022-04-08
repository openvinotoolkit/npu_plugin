//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --transpose-to-permute-cast %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
func @Transpose_d0_d3_d1_d2(%arg0: tensor<1x16x32x64xf16>) -> tensor<1x16x14x30xf16> {
    %cst = const.Declare tensor<16x64x3x3xf16> = dense<1.0> : tensor<16x64x3x3xf16>
    %0 = IE.Transpose(%arg0) {order_value = #map} : tensor<1x16x32x64xf16> -> tensor<1x64x16x32xf16>
    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x64x16x32xf16>, tensor<16x64x3x3xf16> -> tensor<1x16x14x30xf16>
    return %1 : tensor<1x16x14x30xf16>

    // CHECK: %[[CST:.*]] = const.Declare tensor<16x64x3x3xf16> = dense<1.000000e+00> : tensor<16x64x3x3xf16>
    // CHECK: %[[PERM_CAST:.*]] = IE.PermuteCast(%arg0) {dst_order = #NHWC, mem_perm = #NCHW} :
    // CHECK-SAME:  tensor<1x16x32x64xf16> -> tensor<1x64x16x32xf16, {order = #NHWC}>

    // CHECK: %[[REORDER:.*]] = IE.Reorder(%[[PERM_CAST]]) {dstOrder = #NCHW} :
    // CHECK-SAME:  tensor<1x64x16x32xf16, {order = #NHWC}> -> tensor<1x64x16x32xf16>

    // CHECK: %[[CONV:.*]] = IE.Convolution(%[[REORDER]], %[[CST]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:  tensor<1x64x16x32xf16>, tensor<16x64x3x3xf16> -> tensor<1x16x14x30xf16>

    // CHECK: return %[[CONV]] : tensor<1x16x14x30xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWHC = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2, d1)>
func @Transpose_d0_d3_d2_d1(%arg0: tensor<1x16x32x64xf16>) -> tensor<1x16x30x14xf16> {
    %cst = const.Declare tensor<16x64x3x3xf16> = dense<1.0> : tensor<16x64x3x3xf16>
    %0 = IE.Transpose(%arg0) {order_value = #NWHC} : tensor<1x16x32x64xf16> -> tensor<1x64x32x16xf16>
    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x64x32x16xf16>, tensor<16x64x3x3xf16> -> tensor<1x16x30x14xf16>
    return %1 : tensor<1x16x30x14xf16>

    // CHECK: %[[CST:.*]] = const.Declare tensor<16x64x3x3xf16> = dense<1.000000e+00> : tensor<16x64x3x3xf16>
    // CHECK: %[[PERM_CAST:.*]] = IE.PermuteCast(%arg0) {dst_order = #NWHC, mem_perm = #NCHW} :
    // CHECK-SAME:  tensor<1x16x32x64xf16> -> tensor<1x64x32x16xf16, {order = #NWHC}>

    // CHECK: %[[REORDER:.*]] = IE.Reorder(%[[PERM_CAST]]) {dstOrder = #NCHW} :
    // CHECK-SAME:  tensor<1x64x32x16xf16, {order = #NWHC}> -> tensor<1x64x32x16xf16>

    // CHECK: %[[CONV:.*]] = IE.Convolution(%[[REORDER]], %[[CST]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:  tensor<1x64x32x16xf16>, tensor<16x64x3x3xf16> -> tensor<1x16x30x14xf16>

    // CHECK: return %[[CONV]] : tensor<1x16x30x14xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
func @Transpose_d0_d2_d1_d3(%arg0: tensor<1x16x32x64xf16>) -> tensor<1x16x14x62xf16> {
    %cst = const.Declare tensor<16x32x3x3xf16> = dense<1.0> : tensor<16x32x3x3xf16>
    %0 = IE.Transpose(%arg0) {order_value = #NHCW} : tensor<1x16x32x64xf16> -> tensor<1x32x16x64xf16>
    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x32x16x64xf16>, tensor<16x32x3x3xf16> -> tensor<1x16x14x62xf16>
    return %1 : tensor<1x16x14x62xf16>

    // CHECK: %[[CST:.*]] = const.Declare tensor<16x32x3x3xf16> = dense<1.000000e+00> : tensor<16x32x3x3xf16>
    // CHECK: %[[PERM_CAST:.*]] = IE.PermuteCast(%arg0) {dst_order = #NHCW, mem_perm = #NCHW} :
    // CHECK-SAME:  tensor<1x16x32x64xf16> -> tensor<1x32x16x64xf16, {order = #NHCW}>

    // CHECK: %[[REORDER:.*]] = IE.Reorder(%[[PERM_CAST]]) {dstOrder = #NCHW} :
    // CHECK-SAME:  tensor<1x32x16x64xf16, {order = #NHCW}> -> tensor<1x32x16x64xf16>

    // CHECK: %[[CONV:.*]] = IE.Convolution(%[[REORDER]], %[[CST]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:  tensor<1x32x16x64xf16>, tensor<16x32x3x3xf16> -> tensor<1x16x14x62xf16>

    // CHECK: return %[[CONV]] : tensor<1x16x14x62xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NWCH = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
func @Transpose_d0_d2_d3_d1(%arg0: tensor<1x16x32x64xf16>) -> tensor<1x16x62x14xf16> {
    %cst = const.Declare tensor<16x32x3x3xf16> = dense<1.0> : tensor<16x32x3x3xf16>
    %0 = IE.Transpose(%arg0) {order_value = #NHWC} : tensor<1x16x32x64xf16> -> tensor<1x32x64x16xf16>
    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x32x64x16xf16>, tensor<16x32x3x3xf16> -> tensor<1x16x62x14xf16>
    return %1 : tensor<1x16x62x14xf16>

    // CHECK: %[[CST:.*]] = const.Declare tensor<16x32x3x3xf16> = dense<1.000000e+00> : tensor<16x32x3x3xf16>
    // CHECK: %[[PERM_CAST:.*]] = IE.PermuteCast(%arg0) {dst_order = #NWCH, mem_perm = #NCHW} :
    // CHECK-SAME:  tensor<1x16x32x64xf16> -> tensor<1x32x64x16xf16, {order = #NWCH}>

    // CHECK: %[[REORDER:.*]] = IE.Reorder(%[[PERM_CAST]]) {dstOrder = #NCHW} :
    // CHECK-SAME:  tensor<1x32x64x16xf16, {order = #NWCH}> -> tensor<1x32x64x16xf16>

    // CHECK: %[[CONV:.*]] = IE.Convolution(%[[REORDER]], %[[CST]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:  tensor<1x32x64x16xf16>, tensor<16x32x3x3xf16> -> tensor<1x16x62x14xf16>

    // CHECK: return %[[CONV]] : tensor<1x16x62x14xf16>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
func @Transpose_d0_d1_d3_d2(%arg0: tensor<1x16x32x64xf16>) -> tensor<1x16x62x30xf16> {
    %cst = const.Declare tensor<16x16x3x3xf16> = dense<1.0> : tensor<16x16x3x3xf16>
    %0 = IE.Transpose(%arg0) {order_value = #NCWH} : tensor<1x16x32x64xf16> -> tensor<1x16x64x32xf16>
    %1 = IE.Convolution(%0, %cst) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x16x64x32xf16>, tensor<16x16x3x3xf16> -> tensor<1x16x62x30xf16>
    return %1 : tensor<1x16x62x30xf16>

    // CHECK: %[[CST:.*]] = const.Declare tensor<16x16x3x3xf16> = dense<1.000000e+00> : tensor<16x16x3x3xf16>
    // CHECK: %[[PERM_CAST:.*]] = IE.PermuteCast(%arg0) {dst_order = #NCWH, mem_perm = #NCHW} :
    // CHECK-SAME:  tensor<1x16x32x64xf16> -> tensor<1x16x64x32xf16, {order = #NCWH}>

    // CHECK: %[[REORDER:.*]] = IE.Reorder(%[[PERM_CAST]]) {dstOrder = #NCHW} :
    // CHECK-SAME:  tensor<1x16x64x32xf16, {order = #NCWH}> -> tensor<1x16x64x32xf16>

    // CHECK: %[[CONV:.*]] = IE.Convolution(%[[REORDER]], %[[CST]])
    // CHECK-SAME:  {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:  tensor<1x16x64x32xf16>, tensor<16x16x3x3xf16> -> tensor<1x16x62x30xf16>

    // CHECK: return %[[CONV]] : tensor<1x16x62x30xf16>
}
