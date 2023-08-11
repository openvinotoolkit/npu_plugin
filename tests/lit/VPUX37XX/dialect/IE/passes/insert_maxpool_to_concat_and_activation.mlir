//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --insert-maxpool-to-concat-activation %s | FileCheck %s

// CHECK-LABEL: @InsertMaxPoolToConcatAntLRelu
func.func @InsertMaxPoolToConcatAntLRelu(%arg0: tensor<1x128x2x32xf16>, %arg1: tensor<1x128x1x32xf16>) -> tensor<1x128x3x32xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x128x2x32xf16>, tensor<1x128x1x32xf16> -> tensor<1x128x3x32xf16>
    %1 = IE.LeakyRelu(%0) {negative_slope = 0.000000e+00 : f64} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>

    return %1 : tensor<1x128x3x32xf16>

    // CHECK:   %[[VAL_0:.*]] = IE.Concat(%arg0, %arg1) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x128x2x32xf16>, tensor<1x128x1x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   %[[VAL_1:.*]] = IE.AvgPool(%[[VAL_0]]) {
    // CHECK-SAME:      exclude_pads,
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   %[[VAL_2:.*]] = IE.LeakyRelu(%[[VAL_1]]) {negative_slope = 0.000000e+00 : f64} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   return %[[VAL_2]] : tensor<1x128x3x32xf16>
}

// -----

// CHECK-LABEL: @InsertMaxPoolToConcatAndClamp
func.func @InsertMaxPoolToConcatAndClamp(%arg0: tensor<1x128x2x32xf16>, %arg1: tensor<1x128x1x32xf16>) -> tensor<1x128x3x32xf16> {
    %0 = IE.Concat(%arg0, %arg1) {static_offsets = [[0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x128x2x32xf16>, tensor<1x128x1x32xf16> -> tensor<1x128x3x32xf16>
    %1 = IE.Clamp(%0) {max = 0.700000e+00 : f64, min = 0.100000e+00 : f64} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>

    return %1 : tensor<1x128x3x32xf16>

    // CHECK:   %[[VAL_0:.*]] = IE.Concat(%arg0, %arg1) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 2, 0]]} : tensor<1x128x2x32xf16>, tensor<1x128x1x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   %[[VAL_1:.*]] = IE.AvgPool(%[[VAL_0]]) {
    // CHECK-SAME:      exclude_pads,
    // CHECK-SAME:      kernel_size = [1, 1],
    // CHECK-SAME:      pads_begin = [0, 0],
    // CHECK-SAME:      pads_end = [0, 0],
    // CHECK-SAME:      rounding_type = #IE.rounding_type<FLOOR>,
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:  } : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   %[[VAL_2:.*]] = IE.Clamp(%[[VAL_1]]) {max = 0.69999999999999996 : f64, min = 1.000000e-01 : f64} : tensor<1x128x3x32xf16> -> tensor<1x128x3x32xf16>
    // CHECK:   return %[[VAL_2]] : tensor<1x128x3x32xf16>
}
