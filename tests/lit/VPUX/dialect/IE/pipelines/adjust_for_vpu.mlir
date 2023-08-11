//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch% compilation-mode=DefaultHW" --adjust-for-vpu %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertConv1DToConv2DPass
func.func @ConvertConv1DToConv2DPass(%arg0: tensor<1x16x64xf16>) -> tensor<1x1x64xf16> {
    %cts = const.Declare tensor<1x16x5xf16> = dense<1.000000e+00> : tensor<1x16x5xf16>
    %0 = IE.Convolution(%arg0, %cts) {dilations = [1], pads_begin = [2], pads_end = [2], strides = [1]} : tensor<1x16x64xf16>, tensor<1x16x5xf16> -> tensor<1x1x64xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x1x64xf16> -> tensor<1x1x64xf16>

    return %1 : tensor<1x1x64xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x16x1x5xf16> = dense<1.000000e+00> : tensor<1x16x5xf16>, [#const.Reshape<[1, 16, 1, 5]>]

    // CHECK: [[VAL0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 16, 1, 64]} : tensor<1x16x64xf16> -> tensor<1x16x1x64xf16>

    // CHECK: [[VAL1:%.*]] = IE.Convolution([[VAL0]], [[CST]])
    // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 2], pads_end = [0, 2],
    // CHECK-SAME:    strides = [1, 1]}
    // CHECK-SAME:    : tensor<1x16x1x64xf16>, tensor<1x16x1x5xf16> -> tensor<1x1x1x64xf16>

    // CHECK: [[VAL2:%.*]] = IE.AffineReshape([[VAL1]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 1, 64]} : tensor<1x1x1x64xf16> -> tensor<1x1x64xf16>

    // CHECK: [[VAL3:%.*]] = IE.ReLU([[VAL2]]) : tensor<1x1x64xf16> -> tensor<1x1x64xf16>
    // CHECK: return [[VAL3]] : tensor<1x1x64xf16>
}

// -----

// CHECK-LABEL: @FusePostOpReluIntoConv
func.func @FusePostOpReluIntoConv(%arg0: tensor<1x16x64x64xf16>) -> tensor<1x1x64x64xf16> {
    %cts = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    %0 = IE.Convolution(%arg0, %cts) {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x16x64x64xf16>, tensor<1x16x1x1xf16> -> tensor<1x1x64x64xf16>

    %1 = IE.ReLU(%0) :
        tensor<1x1x64x64xf16> -> tensor<1x1x64x64xf16>

    return %1 : tensor<1x1x64x64xf16>

    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<1.000000e+00> : tensor<1x16x1x1xf16>
    // CHECK: [[VAL1:%.*]] = IE.Convolution(%arg0, [[CST]])
    // CHECK-SAME:    {dilations = [1, 1], pads_begin = [0, 0], pads_end = [0, 0],
    // CHECK-SAME:    post_op = {attrs = {}, name = "IE.ReLU"}, strides = [1, 1]}
    // CHECK-SAME:    : tensor<1x16x64x64xf16>, tensor<1x16x1x1xf16> -> tensor<1x1x64x64xf16>

    // CHECK: return [[VAL1]] : tensor<1x1x64x64xf16>
}

// -----

// CHECK-LABEL: @NearestWithSIZESModeConvertToTileOp
func.func @NearestWithSIZESModeConvertToTileOp(%arg0: tensor<1x96x1x1xf32>) -> tensor<1x96x33x33xf32> {
    %0 = IE.Interpolate(%arg0)
        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>,
        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]>
        : vector<4xi32>, scales_attr = [3.300000e+01, 3.300000e+01], sizes_attr = [33, 33]} : tensor<1x96x1x1xf32> -> tensor<1x96x33x33xf32>

    return %0 : tensor<1x96x33x33xf32>

    // CHECK-NOT:   IE.Interpolate
    // CHECK-NOT:   IE.Broadcast
    // CHECK:       [[TILE:%.*]] = IE.Tile(%arg0) {repeats_values = [1, 1, 33, 33]} : tensor<1x96x1x1xf32> -> tensor<1x96x33x33xf32>
    // CHECK:       return [[TILE]] : tensor<1x96x33x33xf32>
}

// -----

// CHECK-LABEL: @NearestWithSCALESModeConvertToTileOp
func.func @NearestWithSCALESModeConvertToTileOp(%arg0: tensor<1x96x1x1xf32>) -> tensor<1x96x33x33xf32> {
    %0 = IE.Interpolate(%arg0)
        {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <ROUND_PREFER_FLOOR>,
        antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]>
        : vector<4xi32>, scales_attr = [3.300000e+01, 3.300000e+01], sizes_attr = [33, 33]} : tensor<1x96x1x1xf32> -> tensor<1x96x33x33xf32>

    return %0 : tensor<1x96x33x33xf32>

    // CHECK-NOT:   IE.Interpolate
    // CHECK-NOT:   IE.Broadcast
    // CHECK:       [[TILE:%.*]] = IE.Tile(%arg0) {repeats_values = [1, 1, 33, 33]} : tensor<1x96x1x1xf32> -> tensor<1x96x33x33xf32>
    // CHECK:       return [[TILE]] : tensor<1x96x33x33xf32>
}
