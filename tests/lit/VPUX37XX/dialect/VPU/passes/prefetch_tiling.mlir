//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX" --prefetch-tiling --canonicalize %s | FileCheck %s

// CHECK-LABEL: func @SplitSwConvOverOC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<256x32x3x3xf16>,
// CHECK-SAME:        [[BIAS:%arg[0-9]]]: tensor<1x256x1x1xf16>
func @SplitSwConvOverOC(
        %input: tensor<1x32x64x64xf16>,
        %filter: tensor<256x32x3x3xf16>,
        %bias: tensor<1x256x1x1xf16>)
            -> tensor<1x256x64x64xf16> {
    %1 = VPU.Convolution(%input, %filter, %bias) {
        dilations = [1, 1],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        strides = [1, 1]
    } : tensor<1x32x64x64xf16>, tensor<256x32x3x3xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x64x64xf16>
    return %1 : tensor<1x256x64x64xf16>

    // Tile 0

    // CHECK:       [[FILTER_TILE0:%.+]] = VPU.Slice [[FILTER]] [0, 0, 0, 0] [128, 32, 3, 3]
    // CHECK-SAME:      : tensor<256x32x3x3xf16> to tensor<128x32x3x3xf16>

    // CHECK:       [[BIAS_TILE0:%.+]] = VPU.Slice [[BIAS]] [0, 0, 0, 0] [1, 128, 1, 1]
    // CHECK-SAME:      : tensor<1x256x1x1xf16> to tensor<1x128x1x1xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Convolution([[INPUT]], [[FILTER_TILE0]], [[BIAS_TILE0]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x128x64x64xf16>

    // Tile 1

    // CHECK:       [[FILTER_TILE1:%.+]] = VPU.Slice [[FILTER]] [128, 0, 0, 0] [128, 32, 3, 3]
    // CHECK-SAME:      : tensor<256x32x3x3xf16> to tensor<128x32x3x3xf16>

    // CHECK:       [[BIAS_TILE1:%.+]] = VPU.Slice [[BIAS]] [0, 128, 0, 0] [1, 128, 1, 1]
    // CHECK-SAME:      : tensor<1x256x1x1xf16> to tensor<1x128x1x1xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Convolution([[INPUT]], [[FILTER_TILE1]], [[BIAS_TILE1]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 2, 1, 1]
    // CHECK-SAME:      -> tensor<1x128x64x64xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 128, 0, 0]
    // CHECK-SAME:      -> tensor<1x256x64x64xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16>
}

// -----

// CHECK-LABEL: func @SplitSwMaxPoolOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x244x244xf16>
func @SplitSwMaxPoolOverH(
        %input: tensor<1x16x244x244xf16>)
            -> tensor<1x16x244x244xf16> {
    %1 = VPU.MaxPool(%input) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = "FLOOR",
        strides = [1, 1]
    } : tensor<1x16x244x244xf16> -> tensor<1x16x244x244xf16>
    return %1 : tensor<1x16x244x244xf16>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 123, 244]
    // CHECK-SAME:       : tensor<1x16x244x244xf16> to tensor<1x16x123x244xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.MaxPool([[INPUT_TILE0]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [1, 1]
    // CHECK-SAME:          pads_end = [0, 1]
    // CHECK-SAME:          rounding_type = "FLOOR"
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      -> tensor<1x16x122x244xf16>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 121, 0] [1, 16, 123, 244]
    // CHECK-SAME:      : tensor<1x16x244x244xf16> to tensor<1x16x123x244xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MaxPool([[INPUT_TILE1]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [0, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = "FLOOR"
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tilingStrategy = [1, 1, 2, 1]
    // CHECK-SAME:      -> tensor<1x16x122x244xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 122, 0]
    // CHECK-SAME:      -> tensor<1x16x244x244xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x244x244xf16>
}

// -----

// CHECK-LABEL: func @SplitSoftMaxOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x20x256x512xf16>
func @SplitSoftMaxOverH(%arg0: tensor<1x20x256x512xf16>) -> tensor<1x20x256x512xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1}: tensor<1x20x256x512xf16> -> tensor<1x20x256x512xf16>
    return %0 : tensor<1x20x256x512xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 20, 43, 512] 
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.SoftMax([[INPUT_TILE0]]) {axisInd = 1 : i64, tilingStrategy = [1, 1, 6, 1]} 
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 43, 0] [1, 20, 43, 512] 
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.SoftMax([[INPUT_TILE1]]) {axisInd = 1 : i64, tilingStrategy = [1, 1, 6, 1]} 
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 86, 0] [1, 20, 43, 512] 
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.SoftMax([[INPUT_TILE2]]) {axisInd = 1 : i64, tilingStrategy = [1, 1, 6, 1]} 
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 129, 0] [1, 20, 43, 512] 
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.SoftMax([[INPUT_TILE3]]) {axisInd = 1 : i64, tilingStrategy = [1, 1, 6, 1]} 
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 172, 0] [1, 20, 42, 512] 
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x42x512xf16>
    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.SoftMax([[INPUT_TILE4]]) {axisInd = 1 : i64, tilingStrategy = [1, 1, 6, 1]} 
    // CHECK-SAME:      : tensor<1x20x42x512xf16> -> tensor<1x20x42x512xf16>

    // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 214, 0] [1, 20, 42, 512] 
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x42x512xf16>
    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.SoftMax([[INPUT_TILE5]]) {axisInd = 1 : i64, tilingStrategy = [1, 1, 6, 1]} 
    // CHECK-SAME:      : tensor<1x20x42x512xf16> -> tensor<1x20x42x512xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]]) 
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]
    // CHECK-SAME:      -> tensor<1x20x256x512xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x20x256x512xf16>
}

// -----

func @InterpSplitOverH(
        %input1: tensor<1x32x64x64xf16>)
            -> tensor<1x32x256x256xf16> {

    %0 = const.Declare tensor<2xsi64> = dense<[256, 256]> : tensor<2xsi64>
    %1 = const.Declare tensor<2xf32>  = dense<[4.000000e+00, 4.00000e+00]> : tensor<2xf32>
    %2 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>

    %3 = VPU.Interpolate(%input1, %0, %1, %2) {
            attr = {antialias = false, coord_mode = "HALF_PIXEL", cube_coeff = -7.500000e-01, mode = "LINEAR", nearest_mode = "ROUND_PREFER_FLOOR", pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = "SIZES"},
            operand_segment_sizes = dense<1> : vector<4xi32> } :
        tensor<1x32x64x64xf16>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x32x256x256xf16>

    return %3 : tensor<1x32x256x256xf16>
}

// CHECK-LABEL: func @InterpSplitOverH
// CHECK-SAME:        [[INPUT1:%arg[0-9]]]: tensor<1x32x64x64xf16>

// Tile 0

// CHECK:       [[TILE0:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 0, 0] [1, 32, 21, 64]
// CHECK-SAME:      : tensor<1x32x64x64xf16> to tensor<1x32x21x64xf16>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 2, 0]
// CHECK-SAME:      : tensor<1x32x21x64xf16>
// CHECK-SAME:      -> tensor<1x32x86x256xf16>

// Tile 1

// CHECK:       [[TILE1:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 21, 0] [1, 32, 21, 64]
// CHECK-SAME:      : tensor<1x32x64x64xf16> to tensor<1x32x21x64xf16>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 1, 0]
// CHECK-SAME:      : tensor<1x32x21x64xf16>
// CHECK-SAME:      -> tensor<1x32x85x256xf16>

// Tile 2

// CHECK:       [[TILE2:%.+]] = VPU.Slice [[INPUT1]] [0, 0, 42, 0] [1, 32, 21, 64]
// CHECK-SAME:      : tensor<1x32x64x64xf16> to tensor<1x32x21x64xf16>
// CHECK:       [[INTERP2:%.+]] = VPU.Interpolate([[TILE2]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 1, 0]
// CHECK-SAME:      : tensor<1x32x21x64xf16>
// CHECK-SAME:      -> tensor<1x32x85x256xf16>

// Concat

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]], [[INTERP2]])
// CHECK-SAME:      [0, 0, 0, 0], [0, 0, 86, 0], [0, 0, 171, 0]
// CHECK-SAME:      -> tensor<1x32x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x32x256x256xf16>

// -----

// CHECK-LABEL: func @SplitPReluOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>
func @SplitPReluOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %cst = const.Declare tensor<1x1x1x8xf16> = dense<[-1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00]> : tensor<8xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[1, 1, 1, 8]>]
    %0 = VPU.PRelu(%arg0, %cst) : tensor<1x8x80x1280xf16>, tensor<1x1x1x8xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[CST:%.+]] = const.Declare tensor<1x1x1x8xf16> = dense<[-1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00]>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.PRelu([[INPUT_TILE0]], [[CST]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16>, tensor<1x1x1x8xf16> -> tensor<1x8x40x1280xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.PRelu([[INPUT_TILE1]], [[CST]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16>, tensor<1x1x1x8xf16> -> tensor<1x8x40x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
    // CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>


  }


// -----

// CHECK-LABEL: func @SplitLeakyReluOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>
func @SplitLeakyReluOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.LeakyRelu(%arg0) {negative_slope = 0.0099999997764825821 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LeakyRelu([[INPUT_TILE0]]) {
    // CHECK-SAME:  negative_slope = 0.0099999997764825821 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LeakyRelu([[INPUT_TILE1]]) {
    // CHECK-SAME:  negative_slope = 0.0099999997764825821 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
    // CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

  }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @GenericTiling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x144x20x20xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<144x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<576x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE1:%arg[0-9]]]: tensor<144x1x1x4xsi32, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE2:%arg[0-9]]]: tensor<576x1x1x4xsi32, {order = #NHWC}>
func @GenericTiling(
        %input: tensor<1x144x20x20xf16, {order = #NHWC}>,
        %weights1: tensor<144x144x3x3xf16, {order = #NHWC}>,
        %weights2: tensor<576x144x3x3xf16, {order = #NHWC}>,
        %weights_table1: tensor<144x1x1x4xsi32, {order = #NHWC}>,
        %weights_table2: tensor<576x1x1x4xsi32, {order = #NHWC}>)
            -> tensor<1x576x20x20xf16, {order = #NHWC}> {
    %1 = VPU.NCE.Convolution(%input, %weights1, %weights_table1) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [144, 144, 3, 3],
        strides = [1, 1]
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<144x144x3x3xf16, {order = #NHWC}>, tensor<144x1x1x4xsi32, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %2 = VPU.NCE.Eltwise(%1, %1) {op_type = "ADD"} : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<1x144x20x20xf16, {order = #NHWC}> -> tensor<1x144x20x20xf16, {order = #NHWC}>
    %3 = VPU.NCE.Convolution(%2, %weights2, %weights_table2) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [576, 144, 3, 3],
        strides = [1, 1]
    } : tensor<1x144x20x20xf16, {order = #NHWC}>, tensor<576x144x3x3xf16, {order = #NHWC}>, tensor<576x1x1x4xsi32, {order = #NHWC}> -> tensor<1x576x20x20xf16, {order = #NHWC}>
    return %3 : tensor<1x576x20x20xf16, {order = #NHWC}>

    // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:     {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [144, 144, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // CHECK:       [[AND:%.+]] = VPU.NCE.Eltwise([[CONV_1]], [[CONV_1]]) {op_type = "ADD"}
    // CHECK-SAME:          -> tensor<1x144x20x20xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[WEIGHTS_TILE0:%.+]] = VPU.Slice [[WEIGHTS2]] [0, 0, 0, 0] [288, 144, 3, 3]
    // CHECK-SAME:      tensor<576x144x3x3xf16, {order = #NHWC}> to tensor<288x144x3x3xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE_TILE0:%.+]] = VPU.Slice [[WEIGHTS_TABLE2]] [0, 0, 0, 0] [288, 1, 1, 4]
    // CHECK-SAME:      tensor<576x1x1x4xsi32, {order = #NHWC}> to tensor<288x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:     {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [288, 144, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:          -> tensor<1x288x20x20xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[WEIGHTS_TILE1:%.+]] = VPU.Slice [[WEIGHTS2]] [288, 0, 0, 0] [288, 144, 3, 3]
    // CHECK-SAME:      tensor<576x144x3x3xf16, {order = #NHWC}> to tensor<288x144x3x3xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE_TILE1:%.+]] = VPU.Slice [[WEIGHTS_TABLE2]] [288, 0, 0, 0] [288, 1, 1, 4]
    // CHECK-SAME:      tensor<576x1x1x4xsi32, {order = #NHWC}> to tensor<288x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:     {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [288, 144, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:          -> tensor<1x288x20x20xf16, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 288, 0, 0]
    // CHECK-SAME:      -> tensor<1x576x20x20xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x576x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32> = dense<1>
    // CHECK-SAME:      : tensor<256x1x1x4xsi32>

    // CHECK:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 64]
    // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_TILE_0]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]}
    // CHECK-SAME:          -> tensor<1x256x32x64xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 64]
    // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_TILE_1]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1], tilingStrategy = [1, 1, 2, 1]}
    // CHECK-SAME:          -> tensor<1x256x32x64xf16, {order = #NHWC}>

    // Concat

    // CHECK:        [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 32, 0]
    // CHECK-SAME:          -> tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEPoolOverH
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x340x340xf16, {order = #NHWC}>)
func @SplitNCEPoolOverH(%arg0: tensor<1x16x340x340xf16, {order = #NHWC}>) -> tensor<1x16x340x340xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %activation_window = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 18 : i64,
        kernel_size = [3, 3],
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        strides = [1, 1]
    } -> tensor<1x16x340x340xf16, {order = #NHWC}>

    return %0 : tensor<1x16x340x340xf16, {order = #NHWC}>

    // CHECK:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>
    // CHECK-SAME:      = dense<1> : tensor<1x1x1x16xui8>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-SAME:      = dense<1> : tensor<16x1x1x4xsi32>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 58, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x58x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:      } -> tensor<1x16x57x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 56, 0] [1, 16, 59, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x59x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x57x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 113, 0] [1, 16, 59, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x59x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE2]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x57x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 170, 0] [1, 16, 59, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x59x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE3]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x57x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 227, 0] [1, 16, 58, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x58x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE4]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x56x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 283, 0] [1, 16, 57, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x57x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE5]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x56x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]]) {
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 57, 0], [0, 0, 114, 0], [0, 0, 171, 0], [0, 0, 228, 0], [0, 0, 284, 0]
    // CHECK-SAME:      -> tensor<1x16x340x340xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x340x340xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoTileWithSOH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {order = #NHWC}>
func @NoTileWithSOH(
        %arg0: tensor<1x32x100x100xf16, {order = #NHWC}>)
            -> tensor<1x128x100x100xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<128x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<128x1x1x4xsi32> = dense<1>
        : tensor<128x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverHeight",
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [128, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x128x100x100xf16, {order = #NHWC}>

    return %0 : tensor<1x128x100x100xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32>
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}>
    // CHECK-NOT:   VPU.Slice

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:          rawFilterShape = [128, 32, 3, 3]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tensor<1x128x100x100xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x128x100x100xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TileWithSOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x16x210x210xf16, {order = #NHWC}>
func @TileWithSOH(
        %arg0: tensor<1x16x210x210xf16, {order = #NHWC}>)
            -> tensor<1x32x210x210xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<32x16x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<32x1x1x4xsi32> = dense<1>
        : tensor<32x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverHeight",
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [32, 16, 3, 3],
        strides = [1, 1]
    } -> tensor<1x32x210x210xf16, {order = #NHWC}>

    return %0 : tensor<1x32x210x210xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}>

    // CHECK:       [[SLICE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 106, 210]
    // CHECK-SAME:          tensor<1x16x210x210xf16, {order = #NHWC}> to tensor<1x16x106x210xf16, {order = #NHWC}>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[SLICE1]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3]
    // CHECK-SAME:          tensor<1x32x105x210xf16, {order = #NHWC}>

    // CHECK:       [[SLICE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 104, 0] [1, 16, 106, 210]
    // CHECK-SAME:          tensor<1x16x210x210xf16, {order = #NHWC}> to tensor<1x16x106x210xf16, {order = #NHWC}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[SLICE2]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverHeight"
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:          rawFilterShape = [32, 16, 3, 3]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tensor<1x32x105x210xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV1]], [[CONV2]])

    // CHECK:       return [[CONCAT]] : tensor<1x32x210x210xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoTileWithSOK
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x10x10xf16, {order = #NHWC}>
func @NoTileWithSOK(
        %arg0: tensor<1x32x10x10xf16, {order = #NHWC}>)
            -> tensor<1x240x10x10xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<240x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<240x32x7x7xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<240x1x1x4xsi32> = dense<1>
        : tensor<240x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverKernel",
        pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64},
        rawFilterShape = [240, 32, 7, 7],
        strides = [1, 1]
    } -> tensor<1x240x10x10xf16, {order = #NHWC}>

    return %0 : tensor<1x240x10x10xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<240x1x1x4xsi32>
    // CHECK:       [[WEIGHTS:%.+]] = const.Declare tensor<240x32x7x7xf16, {order = #NHWC}>
    // CHECK-NOT:   VPU.Slice

    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64},
    // CHECK-SAME:          rawFilterShape = [240, 32, 7, 7],
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:          tensor<1x240x10x10xf16, {order = #NHWC}>

    // CHECK:       return [[CONV]] : tensor<1x240x10x10xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @TileWithSOK
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x30x30xf16, {order = #NHWC}>
func @TileWithSOK(
        %arg0: tensor<1x32x30x30xf16, {order = #NHWC}>)
            -> tensor<1x768x30x30xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<768x32x7x7xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<768x32x7x7xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<768x1x1x4xsi32> = dense<1>
        : tensor<768x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverKernel",
        pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64},
        rawFilterShape = [768, 32, 7, 7],
        strides = [1, 1]
    } -> tensor<1x768x30x30xf16, {order = #NHWC}>

    return %0 : tensor<1x768x30x30xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<384x1x1x4xsi32>
    // CHECK-SAME:          #const.SubView<[384, 0, 0, 0], [384, 1, 1, 4]>]
    // CHECK:       [[WEIGHTS2:%.+]] = const.Declare tensor<384x32x7x7xf16, {order = #NHWC}>
    // CHECK-SAME:          #const.SubView<[384, 0, 0, 0], [384, 32, 7, 7]>
    // CHECK:       [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<384x1x1x4xsi32>
    // CHECK-SAME:          #const.SubView<[0, 0, 0, 0], [384, 1, 1, 4]>
    // CHECK:       [[WEIGHTS1:%.+]] = const.Declare tensor<384x32x7x7xf16, {order = #NHWC}>
    // CHECK-SAME:          #const.SubView<[0, 0, 0, 0], [384, 32, 7, 7]>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64}
    // CHECK-SAME:          rawFilterShape = [384, 32, 7, 7]
    // CHECK-SAME:          tensor<1x384x30x30xf16, {order = #NHWC}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64}
    // CHECK-SAME:          rawFilterShape = [384, 32, 7, 7]
    // CHECK-SAME:          tensor<1x384x30x30xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV1]], [[CONV2]])

    // CHECK:       return [[CONCAT]] : tensor<1x768x30x30xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @LargeConstPipeliningSOKFor
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x256x14x14xf16, {order = #NHWC}>
func @LargeConstPipeliningSOKFor(
        %arg0: tensor<1x256x14x14xf16, {order = #NHWC}>)
            -> tensor<1x512x14x14xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<512x256x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
        : tensor<512x256x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<512x1x1x4xsi32> = dense<1>
        : tensor<512x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        multiClusterStrategy = "SplitOverKernel",
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [512, 256, 3, 3],
        strides = [1, 1]
    } -> tensor<1x512x14x14xf16, {order = #NHWC}>

    return %0 : tensor<1x512x14x14xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-SAME:          [#const.SubView<[256, 0, 0, 0], [256, 1, 1, 4]>]
    // CHECK:       [[WEIGHTS2:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}>
    // CHECK-SAME:          [#const.Reorder<#NHWC>, #const.SubView<[256, 0, 0, 0], [256, 256, 3, 3]>]
    // CHECK:       [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-SAME:          [#const.SubView<[0, 0, 0, 0], [256, 1, 1, 4]>]
    // CHECK:       [[WEIGHTS1:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}>
    // CHECK-SAME:          [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [256, 256, 3, 3]>]

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:          rawFilterShape = [256, 256, 3, 3]
    // CHECK-SAME:          -> tensor<1x256x14x14xf16, {order = #NHWC}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:          rawFilterShape = [256, 256, 3, 3]
    // CHECK-SAME:          -> tensor<1x256x14x14xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV1]], [[CONV2]])

    // CHECK:       return [[CONCAT]] : tensor<1x512x14x14xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @SplitNCEEltwise
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x512x28x28xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x512x28x28xf16, {order = #NHWC}>
func @SplitNCEEltwise(
        %arg0: tensor<1x512x28x28xf16, {order = #NHWC}>,
        %arg1: tensor<1x512x28x28xf16, {order = #NHWC}>)
            -> tensor<1x512x28x28xf16, {order = #NHWC}> {
    %0 = VPU.NCE.Eltwise(%arg0, %arg1) {
        op_type = "ADD"
    } -> tensor<1x512x28x28xf16, {order = #NHWC}>

    return %0 : tensor<1x512x28x28xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:       [[INPUT_0_0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 256, 28, 28]
    // CHECK-SAME:      : tensor<1x512x28x28xf16, {order = #NHWC}> to tensor<1x256x28x28xf16, {order = #NHWC}>
    // CHECK:       [[INPUT_1_0:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 256, 28, 28]
    // CHECK-SAME:      : tensor<1x512x28x28xf16, {order = #NHWC}> to tensor<1x256x28x28xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE_0:%.+]] = VPU.NCE.Eltwise([[INPUT_0_0]], [[INPUT_1_0]])
    // CHECK-SAME:      {op_type = "ADD", tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:      -> tensor<1x256x28x28xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:       [[INPUT_0_1:%.+]] = VPU.Slice [[INPUT_0]] [0, 256, 0, 0] [1, 256, 28, 28]
    // CHECK-SAME:      : tensor<1x512x28x28xf16, {order = #NHWC}> to tensor<1x256x28x28xf16, {order = #NHWC}>
    // CHECK:       [[INPUT_1_1:%.+]] = VPU.Slice [[INPUT_1]] [0, 256, 0, 0] [1, 256, 28, 28]
    // CHECK-SAME:      : tensor<1x512x28x28xf16, {order = #NHWC}> to tensor<1x256x28x28xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[INPUT_0_1]], [[INPUT_1_1]])
    // CHECK-SAME:      {op_type = "ADD", tilingStrategy = [1, 2, 1, 1]}
    // CHECK-SAME:      -> tensor<1x256x28x28xf16, {order = #NHWC}>

    // Concat
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_0]], [[ELTWISE_1]])
    // CHECK-SAME:      : tensor<1x256x28x28xf16, {order = #NHWC}>, tensor<1x256x28x28xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x512x28x28xf16, {order = #NHWC}>

    // return [[CONCAT]] : tensor<1x512x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @NoPrefetchingForEltwise
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x32x70x70xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x64x70x70xf16, {order = #NHWC}>
func @NoPrefetchingForEltwise(
        %arg0: tensor<1x32x70x70xf16, {order = #NHWC}>,
        %arg1: tensor<1x64x70x70xf16, {order = #NHWC}>)
            -> tensor<1x64x70x70xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<64x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<64x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [64, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x64x70x70xf16, {order = #NHWC}>

    %1 = VPU.NCE.Eltwise(%0, %arg1) {
        op_type = "ADD"
    } -> tensor<1x64x70x70xf16, {order = #NHWC}>

    return %1 : tensor<1x64x70x70xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1>
    // CHECK:       [[WEIGHTS:%.+]]       = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>

    // CHECK:       [[PARENT_CONV:%.+]] = VPU.NCE.Convolution([[INPUT_0]], [[WEIGHTS]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          -> tensor<1x64x70x70xf16, {order = #NHWC}>

    // Eltwise is not tiled for prefetching
    // CHECK-NOT:   VPU.Slice
    // CHECK:       [[ELTWISE:%.+]] = VPU.NCE.Eltwise([[PARENT_CONV]], [[INPUT_1]]) {op_type = "ADD"}
    // CHECK-SAME:          -> tensor<1x64x70x70xf16, {order = #NHWC}>

    // return [[ELTWISE]] : tensor<1x64x70x70xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitSparseNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x80x80xf16, {order = #NHWC}>
func @SplitSparseNCEConvOverOH(%arg0: tensor<1x32x80x80xf16, {order = #NHWC}>) -> tensor<1x160x80x80xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]
    %weights_sm = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00> : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]
    %weights_sparse = VPU.GroupSparseTensor(%weights, %weights_sm) {is_weights}
        -> !VPU.SparseTensor<data=tensor<160x32x3x3xf16, {order = #NHWC}>, sparsity_map=tensor<160x1x1x384xi1>, is_weights>
    %weights_table = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<160x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights_sparse, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [160, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x160x80x80xf16, {order = #NHWC}>

    return %0 : tensor<1x160x80x80xf16, {order = #NHWC}>

    // CHECK:        [[WEIGHTS_TABLE_TILE:%.+]] = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10>
    // CHECK-SAME:      : tensor<160x1x1x4xsi32>

    // CHECK:        [[WEIGHTS_TILE:%.+]] = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.GetSparsityMap]

    // CHECK:        [[WEIGHTS_SM_TILE:%.+]] = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<160x32x3x3xf16>, [#const.Reorder<#NHWC>, #const.Sparsify<false>]

    // CHECK:        [[WEIGHTS_SPARSE_TILE:%.+]] = VPU.GroupSparseTensor([[WEIGHTS_SM_TILE]], [[WEIGHTS_TILE]]) {is_weights} -> !VPU.SparseTensor<
    // CHECK-SAME:       data=tensor<160x32x3x3xf16, {order = #NHWC}>,
    // CHECK-SAME:       sparsity_map=tensor<160x1x1x384xi1>, is_weights

    // CHECK:        [[ACTIVATION_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 41, 80]
    // CHECK-SAME:      : tensor<1x32x80x80xf16, {order = #NHWC}> to tensor<1x32x41x80xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_1]], [[WEIGHTS_SPARSE_TILE]], [[WEIGHTS_TABLE_TILE]])
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x160x40x80xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 39, 0] [1, 32, 41, 80]
    // CHECK-SAME:      : tensor<1x32x80x80xf16, {order = #NHWC}> to tensor<1x32x41x80xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_2]], [[WEIGHTS_SPARSE_TILE]], [[WEIGHTS_TABLE_TILE]])
    // CHECK-SAME:          pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [160, 32, 3, 3],
    // CHECK-SAME:          -> tensor<1x160x40x80xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 0, 40, 0]
    // CHECK-SAME:          -> tensor<1x160x80x80xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x160x80x80xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x7x11520xf16, {order = #NHWC}>
func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x11520xf16, {order = #NHWC}>) -> tensor<1x16x1x11520xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_scale = [2.500000e-01]}, strides = [1, 1]} -> tensor<1x16x1x11520xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x11520xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 7, 3840]
    // CHECK-SAME:      tensor<1x16x7x11520xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x3840xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE0]]) {kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x16x1x3840xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 3840] [1, 16, 7, 3840]
    // CHECK-SAME:      tensor<1x16x7x11520xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x3840xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE1]]) {kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x16x1x3840xf16, {order = #NHWC}>

    // Tile 2

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 7680] [1, 16, 7, 3840]
    // CHECK-SAME:      tensor<1x16x7x11520xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x3840xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE2]]) {kernel_size = [7, 1]
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x16x1x3840xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 3840], [0, 0, 0, 7680]
    // CHECK-SAME:      -> tensor<1x16x1x11520xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x11520xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x7x368640xf16, {order = #NHWC}>
func @SplitAveragePoolOverW(%arg0: tensor<1x1x7x368640xf16, {order = #NHWC}>) -> tensor<1x1x1x368640xf16, {order = #NHWC}> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = "FLOOR", strides = [1, 1]} : tensor<1x1x7x368640xf16, {order = #NHWC}> -> tensor<1x1x1x368640xf16, {order = #NHWC}>

    return %0 : tensor<1x1x1x368640xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 7, 122880]
    // CHECK-SAME:      : tensor<1x1x7x368640xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1x7x122880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.AvgPool([[INPUT_TILE0]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x1x1x122880xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 122880] [1, 1, 7, 122880]
    // CHECK-SAME:      : tensor<1x1x7x368640xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1x7x122880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.AvgPool([[INPUT_TILE1]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x1x1x122880xf16, {order = #NHWC}>

    // Tile 2

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 245760] [1, 1, 7, 122880]
    // CHECK-SAME:      : tensor<1x1x7x368640xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1x7x122880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.AvgPool([[INPUT_TILE2]])
    // CHECK-SAME:      tilingStrategy = [1, 1, 1, 3]}
    // CHECK-SAME:      -> tensor<1x1x1x122880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 122880], [0, 0, 0, 245760]
    // CHECK-SAME:      -> tensor<1x1x1x368640xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x1x368640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @SplitOverWForSOHCompatibility
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x8x10001xf16, {order = #NHWC}>
func @SplitOverWForSOHCompatibility(%arg0: tensor<1x16x8x10001xf16, {order = #NHWC}>) -> tensor<1x16x8x10001xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %activation_window = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 4 : i64,
        kernel_size = [1, 1],
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        strides = [1, 1],
        multiClusterStrategy = "SplitOverHeight"
    } -> tensor<1x16x8x10001xf16, {order = #NHWC}>

    return %0 : tensor<1x16x8x10001xf16, {order = #NHWC}>

    // CHECK:       [[ACT_WIN:%.+]] = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>
    // CHECK:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // Tile 0
    // CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 8, 5001]
    // CHECK-SAME:      tensor<1x16x8x10001xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x8x5001xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_0:%.+]] = VPU.NCE.MaxPool([[SLICE_0]], [[WEIGHTS_TABLE]], [[ACT_WIN]])
    // CHECK-SAME:       multiClusterStrategy = "SplitOverHeight",
    // CHECK-SAME:       tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME:      -> tensor<1x16x8x5001xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 5001] [1, 16, 8, 5000]
    // CHECK-SAME:      tensor<1x16x8x10001xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x8x5000xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_1:%.+]] = VPU.NCE.MaxPool([[SLICE_1]], [[WEIGHTS_TABLE]], [[ACT_WIN]])
    // CHECK-SAME:       multiClusterStrategy = "SplitOverHeight",
    // CHECK-SAME:       tilingStrategy = [1, 1, 1, 2]}
    // CHECK-SAME:      -> tensor<1x16x8x5000xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[MAXPOOL_0]], [[MAXPOOL_1]])
    // CHECK:       return [[CONCAT]] : tensor<1x16x8x10001xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func @PrefetchingNCEToUPATask
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x80x80xf16>
func @PrefetchingNCEToUPATask(%arg0: tensor<1x64x80x80xf16>) -> tensor<1x48x80x80xf16, {order = #NHWC}> {
    %weights_0 = const.Declare tensor<48x64x5x5xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<48x64x5x5xf16>, [#const.Reorder<#NHWC>]
    %weights_table_0 = const.Declare tensor<48x1x1x4xsi32> = dense<1> : tensor<48x1x1x4xsi32>
    %0 = VPU.MemPermute(%arg0) {
        dst_order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>,
        mem_perm = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
        } : tensor<1x64x80x80xf16> -> tensor<1x64x80x80xf16, {order = #NHWC}>
    %1 = VPU.NCE.Convolution(%0, %weights_0, %weights_table_0) {
        pad = {bottom = 2 : i64, left = 2 : i64, right = 2 : i64, top = 2 : i64},
        rawFilterShape = [48, 64, 5, 5],
        strides = [1, 1]
    } -> tensor<1x48x80x80xf16, {order = #NHWC}>

    return %1 : tensor<1x48x80x80xf16, {order = #NHWC}>

    // CHECK:       [[WT_2:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1>
    // CHECK-SAME:      : tensor<48x1x1x4xsi32>, [#const.SubView<[32, 0, 0, 0], [16, 1, 1, 4]>]
    // CHECK:       [[WEIGHTS_2:%.+]] = const.Declare tensor<16x64x5x5xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<48x64x5x5xf16>, [#const.Reorder<#NHWC>, #const.SubView<[32, 0, 0, 0], [16, 64, 5, 5]>]
    // CHECK:       [[WT_1:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1>
    // CHECK-SAME:      : tensor<48x1x1x4xsi32>, [#const.SubView<[16, 0, 0, 0], [16, 1, 1, 4]>]
    // CHECK:       [[WEIGHTS_1:%.+]] = const.Declare tensor<16x64x5x5xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<48x64x5x5xf16>, [#const.Reorder<#NHWC>, #const.SubView<[16, 0, 0, 0], [16, 64, 5, 5]>]
    // CHECK:       [[WT_0:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1>
    // CHECK-SAME:      : tensor<48x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [16, 1, 1, 4]>]
    // CHECK:       [[WEIGHTS_0:%.+]] = const.Declare tensor<16x64x5x5xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<48x64x5x5xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [16, 64, 5, 5]>]

    // CHECK:       [[MEMPERMUTE:%.+]] = VPU.MemPermute([[INPUT]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x64x80x80xf16> -> tensor<1x64x80x80xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:       [[CONV_0:%.+]] = VPU.NCE.Convolution([[MEMPERMUTE]], [[WEIGHTS_0]], [[WT_0]]) {
    // CHECK-SAME:      pad = {bottom = 2 : i64, left = 2 : i64, right = 2 : i64, top = 2 : i64},
    // CHECK-SAME:      rawFilterShape = [16, 64, 5, 5],
    // CHECK-SAME:      strides = [1, 1],
    // CHECK-SAME:      tilingStrategy = [1, 3, 1, 1]}
    // CHECK-SAME:      -> tensor<1x16x80x80xf16, {order = #NHWC}>
    // Tile 1
    // CHECK:       [[CONV_1:%.+]] = VPU.NCE.Convolution([[MEMPERMUTE]], [[WEIGHTS_1]], [[WT_1]]) {
    // CHECK-SAME:      pad = {bottom = 2 : i64, left = 2 : i64, right = 2 : i64, top = 2 : i64},
    // CHECK-SAME:      rawFilterShape = [16, 64, 5, 5],
    // CHECK-SAME:      strides = [1, 1],
    // CHECK-SAME:      tilingStrategy = [1, 3, 1, 1]}
    // CHECK-SAME:      -> tensor<1x16x80x80xf16, {order = #NHWC}>
    // Tile 2
    // CHECK:       [[CONV_2:%.+]] = VPU.NCE.Convolution([[MEMPERMUTE]], [[WEIGHTS_2]], [[WT_2]]) {
    // CHECK-SAME:      pad = {bottom = 2 : i64, left = 2 : i64, right = 2 : i64, top = 2 : i64},
    // CHECK-SAME:      rawFilterShape = [16, 64, 5, 5],
    // CHECK-SAME:      strides = [1, 1],
    // CHECK-SAME:      tilingStrategy = [1, 3, 1, 1]}
    // CHECK-SAME:      -> tensor<1x16x80x80xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV_0]], [[CONV_1]], [[CONV_2]])
    // CHECK:       return [[CONCAT]]
}

// -----

// CHECK-LABEL: @ClampSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func @ClampSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Clamp([[INPUT_TILE0]]) {
// CHECK-SAME:  max = 1.000000e+00 : f64, min = -1.000000e+00 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Clamp([[INPUT_TILE1]]) {
// CHECK-SAME:  max = 1.000000e+00 : f64, min = -1.000000e+00 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @ReLUSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func @ReLUSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.ReLU(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.ReLU([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.ReLU([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LogSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func @LogSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Log(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Log([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Log([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @AbsSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func @AbsSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Abs(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Abs([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Abs([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

func @SplitFloorModEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.FloorMod(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitFloorModEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.FloorMod([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.FloorMod([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func @SplitPowerEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitPowerEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Power([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Power([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func @SplitLogicalOrEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.LogicalOr(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitLogicalOrEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LogicalOr([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LogicalOr([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func @SplitLogicalXorEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.LogicalXor(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitLogicalXorEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LogicalXor([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LogicalXor([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func @SplitEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Equal([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Equal([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func @SplitNotEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.NotEqual(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitNotEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NotEqual([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NotEqual([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func @SplitSoftPlusActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.SoftPlus(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SplitSoftPlusActivationSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 8, 40, 1280] 
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.SoftPlus([[INPUT_TILE0]]) { 
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.SoftPlus([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @SplitLessEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitLessEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Less([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Less([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func @SplitLessEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.LessEqual(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitLessEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LessEqual([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LessEqual([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func @SplitGreaterEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitGreaterEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Greater([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Greater([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func @SplitGreaterEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.GreaterEqual(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitGreaterEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.GreaterEqual([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.GreaterEqual([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SplitErfOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func @SplitErfOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Erf(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Erf([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Erf([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @SplitFloorOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func @SplitFloorOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Floor(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Floor([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Floor([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TanSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func @TanSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Tan(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tan([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tan([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @SwishSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func @SwishSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Swish(%arg0) {beta_value = 1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Swish([[INPUT_TILE0]]) {
// CHECK-SAME:  beta_value = 1.000000e+00 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Swish([[INPUT_TILE1]]) {
// CHECK-SAME:  beta_value = 1.000000e+00 : f64, tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

// CHECK-LABEL: @HSigmoidSplitOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func @HSigmoidSplitOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.HSigmoid(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.HSigmoid([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.HSigmoid([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>


}

// -----

func @SplitNegativeActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Negative(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SplitNegativeActivationSw
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:   : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Negative([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:   : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Negative([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @SplitCeilingActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Ceiling(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK-LABEL: @SplitCeilingActivationSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 8, 40, 1280]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Ceiling([[INPUT_TILE0]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 40, 0] [1, 8, 40, 1280]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Ceiling([[INPUT_TILE1]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
    // CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>
}

// -----

func @SplitSignActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Sign(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SplitSignActivationSw
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:   : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Sign([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:   : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Sign([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @SplitSelectEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>, %arg2: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Select(%arg0, %arg1, %arg2) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitSelectEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_2:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 86, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x86x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 86, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x86x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_2]] [0, 0, 0, 0] [1, 10, 86, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x86x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Select([[INPUT_TILE0]], [[INPUT_TILE1]], [[INPUT_TILE2]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 3, 1]} : tensor<1x10x86x256xf16>, tensor<1x10x86x256xf16>, tensor<1x10x86x256xf16> -> tensor<1x10x86x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 86, 0] [1, 10, 85, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x85x256xf16>

// CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 86, 0] [1, 10, 85, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x85x256xf16>

// CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT_2]] [0, 0, 86, 0] [1, 10, 85, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x85x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Select([[INPUT_TILE3]], [[INPUT_TILE4]], [[INPUT_TILE5]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 3, 1]} : tensor<1x10x85x256xf16>, tensor<1x10x85x256xf16>, tensor<1x10x85x256xf16> -> tensor<1x10x85x256xf16>

// CHECK:       [[INPUT_TILE6:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 171, 0] [1, 10, 85, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x85x256xf16>

// CHECK:       [[INPUT_TILE7:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 171, 0] [1, 10, 85, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x85x256xf16>

// CHECK:       [[INPUT_TILE8:%.+]] = VPU.Slice [[INPUT_2]] [0, 0, 171, 0] [1, 10, 85, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x85x256xf16>

// CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.Select([[INPUT_TILE6]], [[INPUT_TILE7]], [[INPUT_TILE8]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 3, 1]} : tensor<1x10x85x256xf16>, tensor<1x10x85x256xf16>, tensor<1x10x85x256xf16> -> tensor<1x10x85x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 86, 0], [0, 0, 171, 0]
// CHECK-SAME:  : tensor<1x10x86x256xf16>, tensor<1x10x85x256xf16>, tensor<1x10x85x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func @SplitAndEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.And(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitAndEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.And([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.And([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func @SplitRoundActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Round(%arg0) {mode = "HALF_TO_EVEN"} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SplitRoundActivationSw
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:   : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Round([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:   : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Round([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @SplitRoundActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Round(%arg0) {mode = "HALF_TO_EVEN"} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SplitRoundActivationSw
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1280]
// CHECK-SAME:   : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Round([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1280]
// CHECK-SAME:   : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Round([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @SplitGeluActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Gelu(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>
}

    // CHECK-LABEL: @SplitGeluActivationSw
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 8, 40, 1280]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Gelu([[INPUT_TILE0]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 40, 0] [1, 8, 40, 1280]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1280xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Gelu([[INPUT_TILE1]]) {
    // CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1280xf16> -> tensor<1x8x40x1280xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
    // CHECK-SAME:  : tensor<1x8x40x1280xf16>, tensor<1x8x40x1280xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func @SplitTopK(%arg0: tensor<1x5x512x512xf16>) -> (tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>) {
    %cst = const.Declare tensor<si32> = dense<1> : tensor<si64>, [#const.ConvertElemType<si32>]
    %output_values, %target_shape = VPU.TopK(%arg0, %cst) {axis = 1 : i64, element_type = si32, mode = "MAX", sort = "SORT_INDICES"} : tensor<1x5x512x512xf16>, tensor<si32> -> tensor<1x1x512x512xf16>, tensor<1x1x512x512xsi32>
    %0 = VPU.Convert(%output_values) {dstElemType = f32} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf32>
    return %0, %target_shape : tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>

    // CHECK-LABEL: @SplitTopK
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x5x512x512xf16>) -> (tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>) {
    // CHECK: [[CST:%.*]] = const.Declare tensor<si32> = dense<1> : tensor<si64>, [#const.ConvertElemType<si32>]
    // CHECK: [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 5, 171, 512] : tensor<1x5x512x512xf16> to tensor<1x5x171x512xf16>
    // CHECK: [[OUTPUT_TILE0:%.+]], [[TARGET_TILE0:%.+]] = VPU.TopK([[INPUT_TILE0]], [[CST]]) {axis = 1 : i64, element_type = si32, mode = "MAX", sort = "SORT_INDICES", tilingStrategy = [1, 1, 3, 1]} : tensor<1x5x171x512xf16>, tensor<si32> -> tensor<1x1x171x512xf16>, tensor<1x1x171x512xsi32>
    // CHECK: [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 171, 0] [1, 5, 171, 512] : tensor<1x5x512x512xf16> to tensor<1x5x171x512xf16>
    // CHECK: [[OUTPUT_TILE1:%.+]], [[TARGET_TILE1:%.+]] = VPU.TopK([[INPUT_TILE1]], [[CST]]) {axis = 1 : i64, element_type = si32, mode = "MAX", sort = "SORT_INDICES", tilingStrategy = [1, 1, 3, 1]} : tensor<1x5x171x512xf16>, tensor<si32> -> tensor<1x1x171x512xf16>, tensor<1x1x171x512xsi32>
    // CHECK: [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 342, 0] [1, 5, 170, 512] : tensor<1x5x512x512xf16> to tensor<1x5x170x512xf16>
    // CHECK: [[OUTPUT_TILE2:%.+]], [[TARGET_TILE2:%.+]] = VPU.TopK([[INPUT_TILE2]], [[CST]]) {axis = 1 : i64, element_type = si32, mode = "MAX", sort = "SORT_INDICES", tilingStrategy = [1, 1, 3, 1]} : tensor<1x5x170x512xf16>, tensor<si32> -> tensor<1x1x170x512xf16>, tensor<1x1x170x512xsi32>
    // CHECK: [[OUTPUT_VALUE:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]]) {static_offsets =
    // CHECK: [[TARGET_SHAPE:%.+]] = VPU.Concat([[TARGET_TILE0]], [[TARGET_TILE1]], [[TARGET_TILE2]]) {static_offsets =
    // CHECK: [[OUTPUT_VALUE_CONV:%.+]] = VPU.Convert([[OUTPUT_VALUE]]) {dstElemType = f32} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf32>
    // CHECK: return [[OUTPUT_VALUE_CONV]], [[TARGET_SHAPE]] : tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>
}

// -----

// CHECK-LABEL: func @SplitStridedSliceOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>
func @SplitStridedSliceOverH(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x640xf16> {
    %0 = VPU.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 8, 80, 1280], new_axis_mask = [0, 0, 0, 0], shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x640xf16>
    return %0 : tensor<1x8x80x640xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 40, 1279]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1279xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.StridedSlice([[INPUT_TILE0]]) {
    // CHECK-SAME:  begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 8, 40, 1279], new_axis_mask = [0, 0, 0, 0], shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2], tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1279xf16> -> tensor<1x8x40x640xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 8, 40, 1279]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x40x1279xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.StridedSlice([[INPUT_TILE1]]) {
    // CHECK-SAME:  begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 8, 40, 1279], new_axis_mask = [0, 0, 0, 0], shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2], tilingStrategy = [1, 1, 2, 1]} : tensor<1x8x40x1279xf16> -> tensor<1x8x40x640xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
    // CHECK-SAME:  : tensor<1x8x40x640xf16>, tensor<1x8x40x640xf16> -> tensor<1x8x80x640xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x640xf16>
  }

// -----

func @SplitLogicalNotEltwiseSw(%arg0: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.LogicalNot(%arg0) : tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitLogicalNotEltwiseSw
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LogicalNot([[INPUT_TILE0]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LogicalNot([[INPUT_TILE1]]) {
// CHECK-SAME:  tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:   @SplitNCEConvOverOCInputOversize
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x32x64xf16, {order = #NHWC}>
func @SplitNCEConvOverOCInputOversize(%arg0: tensor<1x512x32x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
  %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
      pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
      rawFilterShape = [256, 512, 3, 3],
      strides = [1, 1]
  } -> tensor<1x256x32x64xf16, {order = #NHWC}>

  return %0 : tensor<1x256x32x64xf16, {order = #NHWC}>

  // CHECK:       %cst = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[192, 0, 0, 0], [64, 1, 1, 4]>]
  // CHECK:       %cst_0 = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[192, 0, 0, 0], [64, 512, 3, 3]>]
  // CHECK:       %cst_1 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[128, 0, 0, 0], [64, 1, 1, 4]>]
  // CHECK:       %cst_2 = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[128, 0, 0, 0], [64, 512, 3, 3]>]
  // CHECK:       %cst_3 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[64, 0, 0, 0], [64, 1, 1, 4]>]
  // CHECK:       %cst_4 = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[64, 0, 0, 0], [64, 512, 3, 3]>]
  // CHECK:       %cst_5 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 4]>]
  // CHECK:       %cst_6 = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [64, 512, 3, 3]>]
  // CHECK:       %0 = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %1 = VPU.NCE.Convolution(%0, %cst_6, %cst_5) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 4, 2, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %2 = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %3 = VPU.NCE.Convolution(%2, %cst_6, %cst_5) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 4, 2, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %4 = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %5 = VPU.NCE.Convolution(%4, %cst_4, %cst_3) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 4, 2, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %6 = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %7 = VPU.NCE.Convolution(%6, %cst_4, %cst_3) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 4, 2, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %8 = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %9 = VPU.NCE.Convolution(%8, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 4, 2, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %10 = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %11 = VPU.NCE.Convolution(%10, %cst_2, %cst_1) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 4, 2, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %12 = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %13 = VPU.NCE.Convolution(%12, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 4, 2, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %14 = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %15 = VPU.NCE.Convolution(%14, %cst_0, %cst) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 4, 2, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %16 = VPU.Concat(%1, %3, %5, %7, %9, %11, %13, %15)
  // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 16, 0], [0, 64, 0, 0], [0, 64, 16, 0], [0, 128, 0, 0], [0, 128, 16, 0], [0, 192, 0, 0], [0, 192, 16, 0]
  // CHECK-SAME:  : tensor<1x64x16x64xf16, {order = #NHWC}>, tensor<1x64x16x64xf16, {order = #NHWC}>, tensor<1x64x16x64xf16, {order = #NHWC}>, tensor<1x64x16x64xf16, {order = #NHWC}>, tensor<1x64x16x64xf16, {order = #NHWC}>, tensor<1x64x16x64xf16, {order = #NHWC}>, tensor<1x64x16x64xf16, {order = #NHWC}>, tensor<1x64x16x64xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
  // CHECK:       return %16 : tensor<1x256x32x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:   @SplitNCEConvOverOHFilterOversize
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x64x64xf16, {order = #NHWC}>
func @SplitNCEConvOverOHFilterOversize(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
  %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
      pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
      rawFilterShape = [256, 512, 3, 3],
      strides = [1, 1]
  } -> tensor<1x256x64x64xf16, {order = #NHWC}>

  return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

  // CHECK:       %cst = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[128, 0, 0, 0], [128, 1, 1, 4]>]
  // CHECK:       %cst_0 = const.Declare tensor<128x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[128, 0, 0, 0], [128, 512, 3, 3]>]
  // CHECK:       %cst_1 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [128, 1, 1, 4]>]
  // CHECK:       %cst_2 = const.Declare tensor<128x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [128, 512, 3, 3]>]
  // CHECK:       %0 = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %1 = VPU.NCE.Convolution(%0, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x8x64xf16, {order = #NHWC}>
  // CHECK:       %2 = VPU.Slice [[INPUT]] [0, 0, 7, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %3 = VPU.NCE.Convolution(%2, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %4 = VPU.Slice [[INPUT]] [0, 0, 14, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %5 = VPU.NCE.Convolution(%4, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %6 = VPU.Slice [[INPUT]] [0, 0, 21, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %7 = VPU.NCE.Convolution(%6, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %8 = VPU.Slice [[INPUT]] [0, 0, 28, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %9 = VPU.NCE.Convolution(%8, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %10 = VPU.Slice [[INPUT]] [0, 0, 35, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %11 = VPU.NCE.Convolution(%10, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %12 = VPU.Slice [[INPUT]] [0, 0, 42, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %13 = VPU.NCE.Convolution(%12, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %14 = VPU.Slice [[INPUT]] [0, 0, 49, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %15 = VPU.NCE.Convolution(%14, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %16 = VPU.Slice [[INPUT]] [0, 0, 56, 0] [1, 512, 8, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x8x64xf16, {order = #NHWC}>
  // CHECK:       %17 = VPU.NCE.Convolution(%16, %cst_2, %cst_1) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %18 = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %19 = VPU.NCE.Convolution(%18, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x8x64xf16, {order = #NHWC}>
  // CHECK:       %20 = VPU.Slice [[INPUT]] [0, 0, 7, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %21 = VPU.NCE.Convolution(%20, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %22 = VPU.Slice [[INPUT]] [0, 0, 14, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %23 = VPU.NCE.Convolution(%22, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %24 = VPU.Slice [[INPUT]] [0, 0, 21, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %25 = VPU.NCE.Convolution(%24, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %26 = VPU.Slice [[INPUT]] [0, 0, 28, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %27 = VPU.NCE.Convolution(%26, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %28 = VPU.Slice [[INPUT]] [0, 0, 35, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %29 = VPU.NCE.Convolution(%28, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %30 = VPU.Slice [[INPUT]] [0, 0, 42, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %31 = VPU.NCE.Convolution(%30, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %32 = VPU.Slice [[INPUT]] [0, 0, 49, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %33 = VPU.NCE.Convolution(%32, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %34 = VPU.Slice [[INPUT]] [0, 0, 56, 0] [1, 512, 8, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x8x64xf16, {order = #NHWC}>
  // CHECK:       %35 = VPU.NCE.Convolution(%34, %cst_0, %cst) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1], tilingStrategy = [1, 2, 9, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %36 = VPU.Concat(%1, %3, %5, %7, %9, %11, %13, %15, %17, %19, %21, %23, %25, %27, %29, %31, %33, %35)
  // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 15, 0], [0, 0, 22, 0], [0, 0, 29, 0], [0, 0, 36, 0], [0, 0, 43, 0], [0, 0, 50, 0], [0, 0, 57, 0], [0, 128, 0, 0], [0, 128, 8, 0], [0, 128, 15, 0], [0, 128, 22, 0], [0, 128, 29, 0], [0, 128, 36, 0], [0, 128, 43, 0], [0, 128, 50, 0], [0, 128, 57, 0]
  // CHECK-SAME:  : tensor<1x128x8x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x8x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}> -> tensor<1x256x64x64xf16, {order = #NHWC}>
  // CHECK:       return %36 : tensor<1x256x64x64xf16, {order = #NHWC}>
}
func @SplitSquaredDiffEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.SquaredDiff(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitSquaredDiffEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.SquaredDiff([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.SquaredDiff([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = "NUMPY", tilingStrategy = [1, 1, 2, 1]} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>
