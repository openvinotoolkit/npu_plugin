//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-batched-conv-to-1n %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#map = affine_map<(d0, d1) -> (d1, d0)>

func @main(%arg0: tensor<5x16x1x1xf16>, %arg1: tensor<4x16x1x1xf16>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<5x16x1x1xf16>, tensor<4x16x1x1xf16> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[IN_RESHAPE:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 5, 16, 1]} : tensor<5x16x1x1xf16> -> tensor<1x5x16x1xf16>
    // CHECK: [[IN_TRANSPOSE:%.*]] = IE.Transpose([[IN_RESHAPE]]) {order_value = #NHCW} : tensor<1x5x16x1xf16> -> tensor<1x16x5x1xf16>

    // CHECK: [[CONV:%.*]] = IE.Convolution([[IN_TRANSPOSE]], %arg1)
    // CHECK-SAME:   {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:   tensor<1x16x5x1xf16>, tensor<4x16x1x1xf16> -> tensor<1x4x5x1xf16>

    // CHECK: [[OUT_TRANSPOSE:%.*]] = IE.Transpose([[CONV]]) {order_value = #NHCW} : tensor<1x4x5x1xf16> -> tensor<1x5x4x1xf16>
    // CHECK: [[OUT_RESHAPE:%.*]] = IE.Reshape([[OUT_TRANSPOSE]]) {shape_value = [5, 4, 1, 1]} : tensor<1x5x4x1xf16> -> tensor<5x4x1x1xf16>

    // CHECK: return [[OUT_RESHAPE]] : tensor<5x4x1x1xf16>
}

// -----

!qElemType0 = type !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = type !quant.uniform<u8:f16:0, {0.956:128, 0.785:128, 0.567:128, 0.785:128}>

func @MixedPrecisionCase(%arg0: tensor<5x16x1x1x!qElemType0>, %arg1: tensor<4x16x1x1x!qElemType1>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<5x16x1x1x!qElemType0>, tensor<4x16x1x1x!qElemType1> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[IN_RESHAPE:%.*]] = IE.Reshape(%arg0) {shape_value = [1, 5, 16, 1]} : tensor<5x16x1x1x!qElemType0> -> tensor<1x5x16x1x!qElemType0>
    // CHECK: [[IN_TRANSPOSE:%.*]] = IE.Transpose([[IN_RESHAPE]]) {order_value = #NHCW} : tensor<1x5x16x1x!qElemType0> -> tensor<1x16x5x1x!qElemType0>

    // CHECK: [[CONV:%.*]] = IE.Convolution([[IN_TRANSPOSE]], %arg1)
    // CHECK-SAME:   {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} :
    // CHECK-SAME:   tensor<1x16x5x1x!qElemType0>, tensor<4x16x1x1x!qElemType1> -> tensor<1x4x5x1xf16>

    // CHECK: [[OUT_TRANSPOSE:%.*]] = IE.Transpose(%2) {order_value = #NHCW} : tensor<1x4x5x1xf16> -> tensor<1x5x4x1xf16>
    // CHECK: [[OUT_RESHAPE:%.*]] = IE.Reshape([[OUT_TRANSPOSE]]) {shape_value = [5, 4, 1, 1]} : tensor<1x5x4x1xf16> -> tensor<5x4x1x1xf16>
    // CHECK: return [[OUT_RESHAPE]] : tensor<5x4x1x1xf16>
}

// -----

!qElemType0 = type !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>
!qElemType1 = type !quant.uniform<u8:f16:0, {0.956:128, 0.785:128, 0.567:128, 0.785:128}>

func @NoChagesPerAxisQuantization(%arg0: tensor<5x8x1x1x!qElemType0>, %arg1: tensor<4x8x1x1x!qElemType1>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<5x8x1x1x!qElemType0>, tensor<4x8x1x1x!qElemType1> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[CONV:%.*]] = IE.Convolution(%arg0, %arg1)
    // CHECK: return [[CONV]] : tensor<5x4x1x1xf16>
}

// -----

func @NoChangesFilterPlaneNotEq1(%arg0: tensor<5x16x1x1xf16>, %arg1: tensor<4x16x2x2xf16>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [1, 1]} : tensor<5x16x1x1xf16>, tensor<4x16x2x2xf16> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[CONV:%.*]] = IE.Convolution(%arg0, %arg1)
    // CHECK: return [[CONV]] : tensor<5x4x1x1xf16>
}
