//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertConstToAttr
func @ConvertConstToAttr(%arg0: tensor<1x3x10x10xf32>) -> tensor<1x3x20x15xf32> {
    %0 = const.Declare tensor<2xsi64> = dense<[20, 15]> : tensor<2xsi64>
    %1 = const.Declare tensor<2xf32>  = dense<[2.000000e+00, 1.500000e+00]> : tensor<2xf32>
    %2 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>
    // CHECK-NOT:   const.Declare
    %3 = IE.Interpolate(%arg0, %0, %1, %2) {attr = {antialias = false, coord_mode = "HALF_PIXEL", cube_coeff = -7.500000e-01, mode = "NEAREST", nearest_mode = "ROUND_PREFER_FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, operand_segment_sizes = dense<1> : vector<4xi32>} : tensor<1x3x10x10xf32>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x3x20x15xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Interpolate(%arg0) {attr = {antialias = false, coord_mode = "HALF_PIXEL", cube_coeff = -7.500000e-01 : f64, mode = "NEAREST", nearest_mode = "ROUND_PREFER_FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"},
    // CHECK-SAME: axes_attr = [2, 3],
    // CHECK-SAME: operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
    // CHECK-SAME: scales_attr = [2.000000e+00, 1.500000e+00],
    // CHECK-SAME: sizes_attr = [20, 15]}
    // CHECK-SAME: tensor<1x3x10x10xf32> -> tensor<1x3x20x15xf32>

    return %3 : tensor<1x3x20x15xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertConstToAttr3InputsSizes
func @ConvertConstToAttr3InputsSizes(%arg0: tensor<1x3x10x10xf32>) -> tensor<1x3x20x15xf32> {
    %0 = const.Declare tensor<4xsi64> = dense<[1, 3, 20, 15]> : tensor<4xsi64>
    %1 = const.Declare tensor<4xf32>  = dense<[1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>
    // CHECK-NOT:   const.Declare
    %2 = IE.Interpolate(%arg0, %0, %1) {attr = {antialias = false, coord_mode = "HALF_PIXEL", cube_coeff = -7.500000e-01, mode = "NEAREST", nearest_mode = "ROUND_PREFER_FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"}, operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>} : tensor<1x3x10x10xf32>, tensor<4xsi64>, tensor<4xf32> -> tensor<1x3x20x15xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Interpolate(%arg0) {attr = {antialias = false, coord_mode = "HALF_PIXEL", cube_coeff = -7.500000e-01 : f64, mode = "NEAREST", nearest_mode = "ROUND_PREFER_FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"},
    // CHECK-SAME: axes_attr = [0, 1, 2, 3],
    // CHECK-SAME: operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
    // CHECK-SAME: scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00],
    // CHECK-SAME: sizes_attr = [1, 3, 20, 15]}
    // CHECK-SAME: tensor<1x3x10x10xf32> -> tensor<1x3x20x15xf32>

    return %2 : tensor<1x3x20x15xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertConstToAttr3InputsScales
func @ConvertConstToAttr3InputsScales(%arg0: tensor<1x3x10x10xf32>) -> tensor<1x3x20x15xf32> {
    %0 = const.Declare tensor<4xsi64> = dense<[1, 1, 1, 1]> : tensor<4xsi64>
    %1 = const.Declare tensor<4xf32>  = dense<[1.000000e+00, 1.000000e+00, 2.000000e+00, 1.500000e+00]> : tensor<4xf32>
    // CHECK-NOT:   const.Declare
    %2 = IE.Interpolate(%arg0, %0, %1) {attr = {antialias = false, coord_mode = "HALF_PIXEL", cube_coeff = -7.500000e-01, mode = "NEAREST", nearest_mode = "ROUND_PREFER_FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SCALES"}, operand_segment_sizes = dense<[1, 1, 1, 0]> : vector<4xi32>} : tensor<1x3x10x10xf32>, tensor<4xsi64>, tensor<4xf32> -> tensor<1x3x20x15xf32>
    // CHECK:       %[[VAL0:.*]] = IE.Interpolate(%arg0) {attr = {antialias = false, coord_mode = "HALF_PIXEL", cube_coeff = -7.500000e-01 : f64, mode = "NEAREST", nearest_mode = "ROUND_PREFER_FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SCALES"},
    // CHECK-SAME: axes_attr = [0, 1, 2, 3],
    // CHECK-SAME: operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
    // CHECK-SAME: scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 1.500000e+00],
    // CHECK-SAME: sizes_attr = [1, 1, 1, 1]}
    // CHECK-SAME: tensor<1x3x10x10xf32> -> tensor<1x3x20x15xf32>

    return %2 : tensor<1x3x20x15xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @Fold
func @Fold(%arg0: tensor<1x3x512x512xf16>) -> tensor<1x3x512x512xf16> {
        %0 = IE.Interpolate(%arg0)
         {attr = {antialias = false, coord_mode = "PYTORCH_HALF_PIXEL", cube_coeff = -7.500000e-01 : f64, mode = "LINEAR_ONNX", nearest_mode = "FLOOR",
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SCALES"}, axes_attr = [2, 3],
         operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [1.000000e+00, 1.000000e+00], sizes_attr = [512, 512]
         } : tensor<1x3x512x512xf16> -> tensor<1x3x512x512xf16>

    return %0 : tensor<1x3x512x512xf16>

    // CHECK-NOT    IE.Interpolate
    // CHECK:       return %arg0 : tensor<1x3x512x512xf16>
}
