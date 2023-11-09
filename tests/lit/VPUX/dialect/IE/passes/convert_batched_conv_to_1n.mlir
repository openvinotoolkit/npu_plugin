//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-batched-conv-to-1n %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

func.func @main(%arg0: tensor<5x16x1x1xf16>, %arg1: tensor<4x16x1x1xf16>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<5x16x1x1xf16>, tensor<4x16x1x1xf16> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[IN_TRANSPOSE:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<5x16x1x1xf16> -> tensor<1x16x5x1xf16>
    // CHECK: [[CONV:%.*]] = IE.Convolution([[IN_TRANSPOSE]], %arg1) 
    // CHECK-SAME:   {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : 
    // CHECK-SAME:   tensor<1x16x5x1xf16>, tensor<4x16x1x1xf16> -> tensor<1x4x5x1xf16>
    // CHECK: [[OUT_TRANSPOSE:%.*]] = IE.Transpose([[CONV]]) {order_value = #map} : tensor<1x4x5x1xf16> -> tensor<5x4x1x1xf16>

    // CHECK: return [[OUT_TRANSPOSE]] : tensor<5x4x1x1xf16>
}

// -----

#HCNW = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

!qElemType0 = !quant.uniform<u8:f16, 1.000000e+00>
!qElemType1 = !quant.uniform<u8:f16:0, {0.956:128, 0.785:128, 0.567:128, 0.785:128}>

func.func @MixedPrecisionCase(%arg0: tensor<5x16x1x1x!qElemType0>, %arg1: tensor<4x16x1x1x!qElemType1>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<5x16x1x1x!qElemType0>, tensor<4x16x1x1x!qElemType1> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[IN_TRANSPOSE:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<5x16x1x1x!qElemType0> -> tensor<1x16x5x1x!qElemType0>
    // CHECK: [[CONV:%.*]] = IE.Convolution([[IN_TRANSPOSE]], %arg1) 
    // CHECK-SAME:   {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : 
    // CHECK-SAME:   tensor<1x16x5x1x!qElemType0>, tensor<4x16x1x1x!qElemType1> -> tensor<1x4x5x1xf16>
    // CHECK: [[OUT_TRANSPOSE:%.*]] = IE.Transpose([[CONV]]) {order_value = #map} : tensor<1x4x5x1xf16> -> tensor<5x4x1x1xf16>

    // CHECK: return [[OUT_TRANSPOSE]] : tensor<5x4x1x1xf16>
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d2, d1, d0, d3)>

func.func @ConvertWithWNotEq1(%arg0: tensor<512x48x1x336xf16>, %arg1: tensor<6x48x1x3xf16>) -> tensor<512x6x1x336xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : tensor<512x48x1x336xf16>, tensor<6x48x1x3xf16> -> tensor<512x6x1x336xf16>
    return %0 : tensor<512x6x1x336xf16>

    // CHECK: [[IN_TRANSPOSE:%.*]] = IE.Transpose(%arg0) {order_value = #map} : tensor<512x48x1x336xf16> -> tensor<1x48x512x336xf16>
    // CHECK: [[CONV:%.*]] = IE.Convolution([[IN_TRANSPOSE]], %arg1) 
    // CHECK-SAME:   {dilations = [1, 1], pads_begin = [0, 1], pads_end = [0, 1], strides = [1, 1]} : 
    // CHECK-SAME:   tensor<1x48x512x336xf16>, tensor<6x48x1x3xf16> -> tensor<1x6x512x336xf16>
    // CHECK: [[OUT_TRANSPOSE:%.*]] = IE.Transpose([[CONV]]) {order_value = #map} : tensor<1x6x512x336xf16> -> tensor<512x6x1x336xf16>

    // CHECK: return [[OUT_TRANSPOSE]] : tensor<512x6x1x336xf16>
}

// -----

!qElemType0 = !quant.uniform<u8:f16:1, {0.956:128, 0.785:128, 0.567:128, 0.785:128, 0.956:128, 0.785:128, 0.567:128, 0.785:128}>
!qElemType1 = !quant.uniform<u8:f16:0, {0.956:128, 0.785:128, 0.567:128, 0.785:128}>

func.func @NoChagesPerAxisQuantization(%arg0: tensor<5x8x1x1x!qElemType0>, %arg1: tensor<4x8x1x1x!qElemType1>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<5x8x1x1x!qElemType0>, tensor<4x8x1x1x!qElemType1> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>

    // CHECK: [[CONV:%.*]] = IE.Convolution(%arg0, %arg1)
    // CHECK: return [[CONV]] : tensor<5x4x1x1xf16>
}

// -----

func.func @NoChangesFilterPlaneNotEq1(%arg0: tensor<5x16x1x1xf16>, %arg1: tensor<4x16x2x2xf16>) -> tensor<5x4x1x1xf16> {
    %0 = IE.Convolution(%arg0, %arg1) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [1, 1], strides = [1, 1]} : tensor<5x16x1x1xf16>, tensor<4x16x2x2xf16> -> tensor<5x4x1x1xf16>
    return %0 : tensor<5x4x1x1xf16>
    
    // CHECK: [[CONV:%.*]] = IE.Convolution(%arg0, %arg1)
    // CHECK: return [[CONV]] : tensor<5x4x1x1xf16>
}
