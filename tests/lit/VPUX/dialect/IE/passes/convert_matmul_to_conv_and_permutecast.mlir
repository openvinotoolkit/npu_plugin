//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-matmul-to-conv %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @Convert3dMatMulToConvAndPermutecast_transpose_b
func @Convert3dMatMulToConvAndPermutecast_transpose_b(%arg0: tensor<64x100x64xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x100x64xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_b} : tensor<64x100x64xf16>, tensor<64x64xf16> -> tensor<64x100x64xf16>

  return %0 : tensor<64x100x64xf16>

    // CHECK:       %[[RESHAPE_1:.*]] = IE.Reshape(%arg1) {shape_value = [64, 64, 1, 1]} : tensor<64x64xf16> -> tensor<64x64x1x1xf16>
    // CHECK:       %[[RESHAPE_2:.*]] = IE.Reshape(%arg0) {shape_value = [1, 64, 100, 64]} : tensor<64x100x64xf16> -> tensor<1x64x100x64xf16>
    // CHECK:       %[[PERMUTE_2:.*]] = IE.PermuteCast(%[[RESHAPE_2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x64x100x64xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
    // CHECK:       %[[CONV:.*]] = IE.Convolution(%[[PERMUTE_2]], %[[RESHAPE_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}
    // CHECK:       %[[PERMUTE_3:.*]] = IE.PermuteCast(%[[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x100x64xf16>
    // CHECK:       %[[RESHAPE_3:.*]] = IE.Reshape(%[[PERMUTE_3]]) {shape_value = [64, 100, 64]} : tensor<1x64x100x64xf16> -> tensor<64x100x64xf16>
    // CHECK:       return %[[RESHAPE_3]] : tensor<64x100x64xf16>
}

// CHECK-LABEL: @Convert3dMatMulToConvAndPermutecast_transpose_a
func @Convert3dMatMulToConvAndPermutecast_transpose_a(%arg0: tensor<64x64x100xf16>, %arg1: tensor<64x64xf16>) -> tensor<64x100x64xf16> {
  %0 = IE.MatMul(%arg0, %arg1) {transpose_a} : tensor<64x64x100xf16>, tensor<64x64xf16> -> tensor<64x100x64xf16>

  return %0 : tensor<64x100x64xf16>

    // CHECK:       %[[TRANSPOSE_1:.*]] = IE.Transpose(%arg0) {order_value = #map0} : tensor<64x64x100xf16> -> tensor<64x100x64xf16>
    // CHECK:       %[[TRANSPOSE_2:.*]] = IE.Transpose(%arg1) {order_value = #map1} : tensor<64x64xf16> -> tensor<64x64xf16>
    // CHECK:       %[[RESHAPE_1:.*]] = IE.Reshape(%[[TRANSPOSE_2]]) {shape_value = [64, 64, 1, 1]} : tensor<64x64xf16> -> tensor<64x64x1x1xf16>
    // CHECK:       %[[RESHAPE_2:.*]] = IE.Reshape(%[[TRANSPOSE_1]]) {shape_value = [1, 64, 100, 64]} : tensor<64x100x64xf16> -> tensor<1x64x100x64xf16>
    // CHECK:       %[[PERMUTE_2:.*]] = IE.PermuteCast(%[[RESHAPE_2]]) {dst_order = #NHWC, mem_perm = #NCHW} : tensor<1x64x100x64xf16> -> tensor<1x64x64x100xf16, {order = #NHWC}>
    // CHECK:       %[[CONV:.*]] = IE.Convolution(%[[PERMUTE_2]], %[[RESHAPE_1]]) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x64x64x100xf16, {order = #NHWC}
    // CHECK:       %[[PERMUTE_3:.*]] = IE.PermuteCast(%[[CONV]]) {dst_order = #NCHW, mem_perm = #NCHW} : tensor<1x64x64x100xf16, {order = #NHWC}> -> tensor<1x64x100x64xf16>
    // CHECK:       %[[RESHAPE_3:.*]] = IE.Reshape(%[[PERMUTE_3]]) {shape_value = [64, 100, 64]} : tensor<1x64x100x64xf16> -> tensor<64x100x64xf16>
    // CHECK:       return %[[RESHAPE_3]] : tensor<64x100x64xf16>
}
