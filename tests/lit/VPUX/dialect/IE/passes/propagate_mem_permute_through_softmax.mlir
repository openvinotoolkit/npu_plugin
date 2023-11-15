//
// Copyright (C) 2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --propagate-mem-permute-through-softmax --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @PropagateMemPermuteThroughSoftmaxWithAxis3
func.func @PropagateMemPermuteThroughSoftmaxWithAxis3(%arg0: tensor<1x4620x4x16xf16, {order = #NHWC}>) -> tensor<1x16x4x4620xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<[[[[0.000000e+00]], [[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]], [[5.000000e+00]], [[6.000000e+00]], [[7.000000e+00]], [[8.000000e+00]], [[9.000000e+00]], [[1.000000e+01]], [[1.100000e+01]], [[1.200000e+01]], [[1.300000e+01]], [[1.400000e+01]], [[1.500000e+01]]]]> : tensor<1x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
    %0 = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x4620x4x16xf16, {order = #NHWC}> -> tensor<1x4620x4x16xf16, {order = #NHWC}>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x4620x4x16xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    %2 = IE.Convolution(%1, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x4x4620xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    return %2 : tensor<1x16x4x4620xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x4620x4x16xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    // CHECK:               [[SOFTMAX:%.*]] = IE.SoftMax([[MEMPERMUTE]]) {axisInd = 1 : i64} : tensor<1x16x4x4620xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[SOFTMAX]], [[CST]])

    // CHECK:               return [[CONV]] : tensor<1x16x4x4620xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @PropagateMemPermuteThroughSoftmaxWithAxis1
func.func @PropagateMemPermuteThroughSoftmaxWithAxis1(%arg0: tensor<1x16x4x4620xf16, {order = #NHCW}>) -> tensor<1x16x4x4620xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<[[[[0.000000e+00]], [[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]], [[5.000000e+00]], [[6.000000e+00]], [[7.000000e+00]], [[8.000000e+00]], [[9.000000e+00]], [[1.000000e+01]], [[1.100000e+01]], [[1.200000e+01]], [[1.300000e+01]], [[1.400000e+01]], [[1.500000e+01]]]]> : tensor<1x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
    %0 = IE.SoftMax(%arg0) {axisInd = 1 : i64} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHCW}>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    %2 = IE.Convolution(%1, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x4x4620xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    return %2 : tensor<1x16x4x4620xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute(%arg0) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    // CHECK:               [[SOFTMAX:%.*]] = IE.SoftMax([[MEMPERMUTE]]) {axisInd = 1 : i64} : tensor<1x16x4x4620xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[SOFTMAX]], [[CST]])

    // CHECK:               return [[CONV]] : tensor<1x16x4x4620xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @DoNotPropagateMemPermuteThroughSoftmax
func.func @DoNotPropagateMemPermuteThroughSoftmax(%arg0: tensor<1x16x4x4620xf16, {order = #NHCW}>) -> tensor<1x16x4x4620xf16, {order = #NHWC}> {
    %cst_0 = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}> = dense<[[[[0.000000e+00]], [[1.000000e+00]], [[2.000000e+00]], [[3.000000e+00]], [[4.000000e+00]], [[5.000000e+00]], [[6.000000e+00]], [[7.000000e+00]], [[8.000000e+00]], [[9.000000e+00]], [[1.000000e+01]], [[1.100000e+01]], [[1.200000e+01]], [[1.300000e+01]], [[1.400000e+01]], [[1.500000e+01]]]]> : tensor<1x16x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reorder<#NHWC>, #const.PadWithZero<[0, 0, 0, 0], [15, 0, 0, 0]>]
    %0 = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHCW}>
    %1 = IE.MemPermute(%0) {dst_order = #NHWC, mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    %2 = IE.Convolution(%1, %cst_0) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x4x4620xf16, {order = #NHWC}>, tensor<16x16x1x1xf16, {order = #NHWC}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    return %2 : tensor<1x16x4x4620xf16, {order = #NHWC}>

    // CHECK-DAG:           [[CST:%.+]] = const.Declare tensor<16x16x1x1xf16, {order = #NHWC}>
    // CHECK:               [[SOFTMAX:%.*]] = IE.SoftMax(%arg0) {axisInd = 3 : i64} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHCW}>
    // CHECK:               [[MEMPERMUTE:%.*]] = IE.MemPermute([[SOFTMAX]]) {dst_order = #NHWC, mem_perm = #NCWH} : tensor<1x16x4x4620xf16, {order = #NHCW}> -> tensor<1x16x4x4620xf16, {order = #NHWC}>
    // CHECK:   [[CONV:%.+]] = IE.Convolution([[MEMPERMUTE]], [[CST]])

    // CHECK:               return [[CONV]] : tensor<1x16x4x4620xf16, {order = #NHWC}>
}
