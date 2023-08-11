//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX37XX compilation-mode=DefaultHW" --tiling="enable-prefetch=true" %s | FileCheck %s

// CHECK-LABEL: func.func @SplitSwConvOverOC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<256x32x3x3xf16>,
// CHECK-SAME:        [[BIAS:%arg[0-9]]]: tensor<1x256x1x1xf16>
func.func @SplitSwConvOverOC(
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
    // CHECK-SAME:      -> tensor<1x128x64x64xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 128, 0, 0]
    // CHECK-SAME:      -> tensor<1x256x64x64xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x64x64xf16>
}

// -----

// CHECK-LABEL: func.func @SplitSwMaxPoolOverH
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x244x244xf16>
func.func @SplitSwMaxPoolOverH(
        %input: tensor<1x16x244x244xf16>)
            -> tensor<1x16x244x244xf16> {
    %1 = VPU.MaxPool(%input) {
        kernel_size = [3, 3],
        pads_begin = [1, 1],
        pads_end = [1, 1],
        rounding_type = #IE.rounding_type<FLOOR>,
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
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x122x244xf16>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 121, 0] [1, 16, 123, 244]
    // CHECK-SAME:      : tensor<1x16x244x244xf16> to tensor<1x16x123x244xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.MaxPool([[INPUT_TILE1]])
    // CHECK-SAME:          kernel_size = [3, 3]
    // CHECK-SAME:          pads_begin = [0, 1]
    // CHECK-SAME:          pads_end = [1, 1]
    // CHECK-SAME:          rounding_type = #IE.rounding_type<FLOOR>
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x122x244xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 122, 0]
    // CHECK-SAME:      -> tensor<1x16x244x244xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x244x244xf16>
}

// -----

// CHECK-LABEL: func.func @SplitSoftMaxOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x20x256x512xf16>
func.func @SplitSoftMaxOverW(%arg0: tensor<1x20x256x512xf16>) -> tensor<1x20x256x512xf16> {
    %0 = VPU.SoftMax(%arg0) {axisInd = 1}: tensor<1x20x256x512xf16> -> tensor<1x20x256x512xf16>
    return %0 : tensor<1x20x256x512xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 20, 256, 86]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x256x86xf16>
    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.SoftMax([[INPUT_TILE0]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x256x86xf16> -> tensor<1x20x256x86xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 86] [1, 20, 256, 86]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x256x86xf16>
    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.SoftMax([[INPUT_TILE1]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x256x86xf16> -> tensor<1x20x256x86xf16>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 172] [1, 20, 256, 85]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x256x85xf16>
    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.SoftMax([[INPUT_TILE2]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x256x85xf16> -> tensor<1x20x256x85xf16>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 257] [1, 20, 256, 85]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x256x85xf16>
    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.SoftMax([[INPUT_TILE3]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x256x85xf16> -> tensor<1x20x256x85xf16>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 342] [1, 20, 256, 85]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x256x85xf16>
    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.SoftMax([[INPUT_TILE4]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x256x85xf16> -> tensor<1x20x256x85xf16>

    // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 427] [1, 20, 256, 85]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x256x85xf16>
    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.SoftMax([[INPUT_TILE5]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x256x85xf16> -> tensor<1x20x256x85xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 86], [0, 0, 0, 172], [0, 0, 0, 257], [0, 0, 0, 342], [0, 0, 0, 427]
    // CHECK-SAME:      -> tensor<1x20x256x512xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x20x256x512xf16>
}

// -----

// CHECK-LABEL: func.func @SplitLogSoftmaxOverH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x20x256x512xf16>
func.func @SplitLogSoftmaxOverH(%arg0: tensor<1x20x256x512xf16>) -> tensor<1x20x256x512xf16> {
    %0 = VPU.LogSoftmax(%arg0) {axisInd = 1}: tensor<1x20x256x512xf16> -> tensor<1x20x256x512xf16>
    return %0 : tensor<1x20x256x512xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 20, 43, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LogSoftmax([[INPUT_TILE0]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 43, 0] [1, 20, 43, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LogSoftmax([[INPUT_TILE1]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 86, 0] [1, 20, 43, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.LogSoftmax([[INPUT_TILE2]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 129, 0] [1, 20, 43, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x43x512xf16>
    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.LogSoftmax([[INPUT_TILE3]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x43x512xf16> -> tensor<1x20x43x512xf16>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 172, 0] [1, 20, 42, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x42x512xf16>
    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.LogSoftmax([[INPUT_TILE4]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x42x512xf16> -> tensor<1x20x42x512xf16>

    // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 214, 0] [1, 20, 42, 512]
    // CHECK-SAME:      : tensor<1x20x256x512xf16> to tensor<1x20x42x512xf16>
    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.LogSoftmax([[INPUT_TILE5]]) {axisInd = 1 : i64}
    // CHECK-SAME:      : tensor<1x20x42x512xf16> -> tensor<1x20x42x512xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 43, 0], [0, 0, 86, 0], [0, 0, 129, 0], [0, 0, 172, 0], [0, 0, 214, 0]
    // CHECK-SAME:      -> tensor<1x20x256x512xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x20x256x512xf16>
}

// -----

// CHECK-LABEL: func.func @InterpSplitOverC
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x24x64x64xf16>
func.func @InterpSplitOverC(
        %input1: tensor<1x24x64x64xf16>)
            -> tensor<1x24x256x256xf16> {

    %0 = const.Declare tensor<2xsi64> = dense<[256, 256]> : tensor<2xsi64>
    %1 = const.Declare tensor<2xf32>  = dense<[4.000000e+00, 4.00000e+00]> : tensor<2xf32>
    %2 = const.Declare tensor<2xsi64> = dense<[2, 3]> : tensor<2xsi64>

    %3 = VPU.Interpolate(%input1, %0, %1, %2) {
            attr = #IE.Interpolate<antialias = false, coord_mode = <HALF_PIXEL>, cube_coeff = -7.500000e-01, mode = <LINEAR>, nearest_mode = <ROUND_PREFER_FLOOR>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>,
            operand_segment_sizes = dense<1> : vector<4xi32> } :
        tensor<1x24x64x64xf16>, tensor<2xsi64>, tensor<2xf32>, tensor<2xsi64> -> tensor<1x24x256x256xf16>

    return %3 : tensor<1x24x256x256xf16>
}

// CHECK:       [[TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 12, 64, 64]
// CHECK-SAME:      : tensor<1x24x64x64xf16> to tensor<1x12x64x64xf16>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x12x64x64xf16>
// CHECK-SAME:      -> tensor<1x12x256x256xf16>
// CHECK:       [[TILE1:%.+]] = VPU.Slice %arg0 [0, 12, 0, 0] [1, 12, 64, 64]
// CHECK-SAME:      : tensor<1x24x64x64xf16> to tensor<1x12x64x64xf16>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x12x64x64xf16>
// CHECK-SAME:      -> tensor<1x12x256x256xf16>
// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]]) {
// CHECK-SAME:      static_offsets = {{\[\[}}0, 0, 0, 0], [0, 12, 0, 0]]}
// CHECK-SAME:      : tensor<1x12x256x256xf16>, tensor<1x12x256x256xf16> -> tensor<1x24x256x256xf16>
// CHECK:       return [[OUTPUT]] : tensor<1x24x256x256xf16>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverCDueToUglyScalingFactor
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x21x65x65xf16, {order = #NHWC}>
func.func @InterpSplitOverCDueToUglyScalingFactor(%arg0: tensor<1x21x65x65xf16, {order = #NHWC}>) -> tensor<1x21x513x513xf16, {order = #NHWC}> {
  %0 = VPU.Interpolate(%arg0) {
    attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode =  <LINEAR_ONNX>, nearest_mode = <SIMPLE>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
    axes_attr = [2, 3],
    multiClusterStrategy = "Clustering",
    operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
    scales_attr = [7.8923077583312988, 7.8923077583312988],
    sizes_attr = [513, 513]} : tensor<1x21x65x65xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}> -> tensor<1x21x513x513xf16, {order = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>}>
  return %0 : tensor<1x21x513x513xf16, {order = #NHWC}>

// CHECK:       [[TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 3, 65, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16, {order = #NHWC}> to tensor<1x3x65x65xf16, {order = #NHWC}>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x3x65x65xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x3x513x513xf16, {order = #NHWC}>
// CHECK:       [[TILE1:%.+]] = VPU.Slice %arg0 [0, 3, 0, 0] [1, 3, 65, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16, {order = #NHWC}> to tensor<1x3x65x65xf16, {order = #NHWC}>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x3x65x65xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x3x513x513xf16, {order = #NHWC}>
// CHECK:       [[TILE2:%.+]] = VPU.Slice %arg0 [0, 6, 0, 0] [1, 3, 65, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16, {order = #NHWC}> to tensor<1x3x65x65xf16, {order = #NHWC}>
// CHECK:       [[INTERP2:%.+]] = VPU.Interpolate([[TILE2]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x3x65x65xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x3x513x513xf16, {order = #NHWC}>
// CHECK:       [[TILE3:%.+]] = VPU.Slice %arg0 [0, 9, 0, 0] [1, 3, 65, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16, {order = #NHWC}> to tensor<1x3x65x65xf16, {order = #NHWC}>
// CHECK:       [[INTERP3:%.+]] = VPU.Interpolate([[TILE3]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x3x65x65xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x3x513x513xf16, {order = #NHWC}>
// CHECK:       [[TILE4:%.+]] = VPU.Slice %arg0 [0, 12, 0, 0] [1, 3, 65, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16, {order = #NHWC}> to tensor<1x3x65x65xf16, {order = #NHWC}>
// CHECK:       [[INTERP4:%.+]] = VPU.Interpolate([[TILE4]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x3x65x65xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x3x513x513xf16, {order = #NHWC}>
// CHECK:       [[TILE5:%.+]] = VPU.Slice %arg0 [0, 15, 0, 0] [1, 3, 65, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16, {order = #NHWC}> to tensor<1x3x65x65xf16, {order = #NHWC}>
// CHECK:       [[INTERP5:%.+]] = VPU.Interpolate([[TILE5]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x3x65x65xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x3x513x513xf16, {order = #NHWC}>
// CHECK:       [[TILE6:%.+]] = VPU.Slice %arg0 [0, 18, 0, 0] [1, 3, 65, 65]
// CHECK-SAME:      : tensor<1x21x65x65xf16, {order = #NHWC}> to tensor<1x3x65x65xf16, {order = #NHWC}>
// CHECK:       [[INTERP6:%.+]] = VPU.Interpolate([[TILE6]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0]
// CHECK-SAME:      : tensor<1x3x65x65xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x3x513x513xf16, {order = #NHWC}>
// CHECK:       [[CONCAT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]], [[INTERP2]], [[INTERP3]], [[INTERP4]], [[INTERP5]], [[INTERP6]]) {
// CHECK-SAME:      static_offsets = {{\[\[}}0, 0, 0, 0], [0, 3, 0, 0], [0, 6, 0, 0], [0, 9, 0, 0], [0, 12, 0, 0], [0, 15, 0, 0], [0, 18, 0, 0]]} : tensor<1x3x513x513xf16, {order = #NHWC}>, tensor<1x3x513x513xf16, {order = #NHWC}>, tensor<1x3x513x513xf16, {order = #NHWC}>, tensor<1x3x513x513xf16, {order = #NHWC}>, tensor<1x3x513x513xf16, {order = #NHWC}>, tensor<1x3x513x513xf16, {order = #NHWC}>, tensor<1x3x513x513xf16, {order = #NHWC}> -> tensor<1x21x513x513xf16, {order = #NHWC}>
// CHECK:       return [[CONCAT]] : tensor<1x21x513x513xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @InterpSplitOverCDueTo1Size
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x256x1x1xf16, {order = #NHWC}>
func.func @InterpSplitOverCDueTo1Size(%arg0: tensor<1x256x1x1xf16, {order = #NHWC}>) -> tensor<1x256x65x65xf16, {order = #NHWC}> {
  %0 = VPU.Interpolate(%arg0) {
    attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode =  <LINEAR_ONNX>, nearest_mode = <SIMPLE>, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>,
    axes_attr = [2, 3],
    multiClusterStrategy = "Clustering",
    operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
    scales_attr = [6.500000e+01, 6.500000e+01],
    sizes_attr = [65, 65]} : tensor<1x256x1x1xf16, {order = #NHWC}> -> tensor<1x256x65x65xf16, {order = #NHWC}>

    return %0 : tensor<1x256x65x65xf16, {order = #NHWC}>

// CHECK:       [[TILE0:%.+]] = VPU.Slice %arg0 [0, 0, 0, 0] [1, 128, 1, 1]
// CHECK-SAME:      : tensor<1x256x1x1xf16, {order = #NHWC}> to tensor<1x128x1x1xf16, {order = #NHWC}>
// CHECK:       [[INTERP0:%.+]] = VPU.Interpolate([[TILE0]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
// CHECK-SAME:      : tensor<1x128x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x128x65x65xf16, {order = #NHWC}>
// CHECK:       [[TILE1:%.+]] = VPU.Slice %arg0 [0, 128, 0, 0] [1, 128, 1, 1]
// CHECK-SAME:      : tensor<1x256x1x1xf16, {order = #NHWC}> to tensor<1x128x1x1xf16, {order = #NHWC}>
// CHECK:       [[INTERP1:%.+]] = VPU.Interpolate([[TILE1]]) {
// CHECK-SAME:      pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
// CHECK-SAME:      : tensor<1x128x1x1xf16, {order = #NHWC}>
// CHECK-SAME:      -> tensor<1x128x65x65xf16, {order = #NHWC}>
// CHECK:       [[CONCAT:%.+]] = VPU.Concat([[INTERP0]], [[INTERP1]]) {
// CHECK-SAME:      static_offsets = {{\[\[}}0, 0, 0, 0], [0, 128, 0, 0]]} : tensor<1x128x65x65xf16, {order = #NHWC}>, tensor<1x128x65x65xf16, {order = #NHWC}> -> tensor<1x256x65x65xf16, {order = #NHWC}>
// CHECK:       return [[CONCAT]] : tensor<1x256x65x65xf16, {order = #NHWC}>
}

// -----

// CHECK-LABEL: func.func @SplitPReluOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>
func.func @SplitPReluOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %cst = const.Declare tensor<1x8x1x1xf16> = dense<[-1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00]> : tensor<8xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[1, 8, 1, 1]>]
    %0 = VPU.PRelu(%arg0, %cst) : tensor<1x8x80x1280xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK-DAG:       [[CST:%.+]] = const.Declare tensor<1x8x1x1xf16> = dense<[-1.000000e+01, -9.000000e+00, -8.000000e+00, -7.000000e+00, -6.000000e+00, -5.000000e+00, -4.000000e+00, -3.000000e+00]>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.PRelu([[INPUT_TILE0]], [[CST]])
    // CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x640xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640]  [1, 8, 80, 640]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.PRelu([[INPUT_TILE1]], [[CST]])
    // CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x1x1xf16> -> tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
    // CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>


  }

// -----

// CHECK-LABEL: func.func @SplitLeakyReluOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>
func.func @SplitLeakyReluOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.LeakyRelu(%arg0) {negative_slope = 0.0099999997764825821 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LeakyRelu([[INPUT_TILE0]]) {
    // CHECK-SAME:  negative_slope = 0.0099999997764825821 : f64} : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LeakyRelu([[INPUT_TILE1]]) {
    // CHECK-SAME:  negative_slope = 0.0099999997764825821 : f64} : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
    // CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

  }

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @GenericTiling
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x144x20x20xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS1:%arg[0-9]]]: tensor<144x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS2:%arg[0-9]]]: tensor<576x144x3x3xf16, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE1:%arg[0-9]]]: tensor<144x1x1x4xsi32, {order = #NHWC}>,
// CHECK-SAME:        [[WEIGHTS_TABLE2:%arg[0-9]]]: tensor<576x1x1x4xsi32, {order = #NHWC}>
func.func @GenericTiling(
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

    // CHECK:       [[WEIGHTS_TILE0:%.+]] = VPU.Slice [[WEIGHTS2]] [0, 0, 0, 0] [192, 144, 3, 3]
    // CHECK-SAME:      tensor<576x144x3x3xf16, {order = #NHWC}> to tensor<192x144x3x3xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE_TILE0:%.+]] = VPU.Slice [[WEIGHTS_TABLE2]] [0, 0, 0, 0] [192, 1, 1, 4]
    // CHECK-SAME:      tensor<576x1x1x4xsi32, {order = #NHWC}> to tensor<192x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:     {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [192, 144, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x192x20x20xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[WEIGHTS_TILE1:%.+]] = VPU.Slice [[WEIGHTS2]] [192, 0, 0, 0] [192, 144, 3, 3]
    // CHECK-SAME:      tensor<576x144x3x3xf16, {order = #NHWC}> to tensor<192x144x3x3xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE_TILE1:%.+]] = VPU.Slice [[WEIGHTS_TABLE2]] [192, 0, 0, 0] [192, 1, 1, 4]
    // CHECK-SAME:      tensor<576x1x1x4xsi32, {order = #NHWC}> to tensor<192x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:     {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [192, 144, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x192x20x20xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[WEIGHTS_TILE2:%.+]] = VPU.Slice [[WEIGHTS2]] [384, 0, 0, 0] [192, 144, 3, 3]
    // CHECK-SAME:      tensor<576x144x3x3xf16, {order = #NHWC}> to tensor<192x144x3x3xf16, {order = #NHWC}>

    // CHECK:       [[WEIGHTS_TABLE_TILE2:%.+]] = VPU.Slice [[WEIGHTS_TABLE2]] [384, 0, 0, 0] [192, 1, 1, 4]
    // CHECK-SAME:      tensor<576x1x1x4xsi32, {order = #NHWC}> to tensor<192x1x1x4xsi32, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.Convolution([[AND]], [[WEIGHTS_TILE2]], [[WEIGHTS_TABLE_TILE2]])
    // CHECK-SAME:     {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [192, 144, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x192x20x20xf16, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 192, 0, 0], [0, 384, 0, 0]
    // CHECK-SAME:      -> tensor<1x576x20x20xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x576x20x20xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL:   @SplitNCEConvOverOH
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOH(%arg0: tensor<1x32x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
    %weights = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x32x3x3xf16>, [#const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        rawFilterShape = [256, 32, 3, 3],
        strides = [1, 1]
    } -> tensor<1x256x64x64xf16, {order = #NHWC}>

    return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-DAG:        [[FILTER:%.+]] = const.Declare tensor<256x32x3x3xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 32, 33, 64]
    // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[ACTIVATION_TILE_0]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
    // CHECK-SAME:          -> tensor<1x256x32x64xf16, {order = #NHWC}>

    // CHECK:        [[ACTIVATION_TILE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 31, 0] [1, 32, 33, 64]
    // CHECK-SAME:      : tensor<1x32x64x64xf16, {order = #NHWC}> to tensor<1x32x33x64xf16, {order = #NHWC}>

    // CHECK:        [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[ACTIVATION_TILE_1]], [[FILTER]], [[WEIGHTS_TABLE]])
    // CHECK-SAME:          {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [256, 32, 3, 3], strides = [1, 1]}
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
func.func @SplitNCEPoolOverH(%arg0: tensor<1x16x340x340xf16, {order = #NHWC}>) -> tensor<1x16x340x340xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %activation_window = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 18 : i64,
        kernel_size = [3, 3],
        pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
        strides = [1, 1]
    } -> tensor<1x16x340x340xf16, {order = #NHWC}>

    return %0 : tensor<1x16x340x340xf16, {order = #NHWC}>

    // CHECK-DAG:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 50, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x50x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}
    // CHECK-SAME:      } -> tensor<1x16x49x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 48, 0] [1, 16, 51, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x51x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x49x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 97, 0] [1, 16, 51, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x51x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE2]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x49x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 146, 0] [1, 16, 51, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x51x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE3]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x49x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 195, 0] [1, 16, 50, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x50x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE4]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x48x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 243, 0] [1, 16, 50, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x50x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE5]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x48x340xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE6:%.+]] = VPU.Slice [[INPUT]] [0, 0, 291, 0] [1, 16, 49, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x49x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE6:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE6]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x48x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]], [[OUTPUT_TILE6]]) {
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 49, 0], [0, 0, 98, 0], [0, 0, 147, 0], [0, 0, 196, 0], [0, 0, 244, 0], [0, 0, 292, 0]
    // CHECK-SAME:      -> tensor<1x16x340x340xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x340x340xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEPoolOverHAndKeepWidthSize
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x340x340xf16, {order = #NHWC}>)
func.func @SplitNCEPoolOverHAndKeepWidthSize(%arg0: tensor<1x16x340x340xf16, {order = #NHWC}>) -> tensor<1x16x170x170xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %activation_window = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 4 : i64,
        kernel_size = [1, 1],
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        strides = [2, 2]
    } -> tensor<1x16x170x170xf16, {order = #NHWC}>

    return %0 : tensor<1x16x170x170xf16, {order = #NHWC}>

    // CHECK-DAG:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 67, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x67x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x34x170xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 68, 0] [1, 16, 67, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x67x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x34x170xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 136, 0] [1, 16, 67, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x67x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE2]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x34x170xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 204, 0] [1, 16, 67, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x67x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE3]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x34x170xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 272, 0] [1, 16, 67, 340]
    // CHECK-SAME:      : tensor<1x16x340x340xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x67x340xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE4]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x34x170xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]]) {
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 34, 0], [0, 0, 68, 0], [0, 0, 102, 0], [0, 0, 136, 0]
    // CHECK-SAME:      -> tensor<1x16x170x170xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x170x170xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitNCEPoolOverWAndKeepHeightSize
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x16x2x80000xf16, {order = #NHWC}>)
func.func @SplitNCEPoolOverWAndKeepHeightSize(%arg0: tensor<1x16x2x80000xf16, {order = #NHWC}>) -> tensor<1x16x1x40000xf16, {order = #NHWC}> {
    %weights_table = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>
    %activation_window = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>

    %0 = VPU.NCE.MaxPool(%arg0, %weights_table, %activation_window) {
        activation_window_channel_length = 4 : i64,
        kernel_size = [1, 1],
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        strides = [2, 2]
    } -> tensor<1x16x1x40000xf16, {order = #NHWC}>

    return %0 : tensor<1x16x1x40000xf16, {order = #NHWC}>

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32>
    // CHECK-DAG:       [[ACTIVATION_WINDOW:%.+]] = const.Declare tensor<1x1x1x16xui8>

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 2, 11429]
    // CHECK-SAME:      : tensor<1x16x2x80000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x2x11429xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE0]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x1x5715xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 11430] [1, 16, 2, 11429]
    // CHECK-SAME:      : tensor<1x16x2x80000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x2x11429xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE1]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x1x5715xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 22860] [1, 16, 2, 11427]
    // CHECK-SAME:      : tensor<1x16x2x80000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x2x11427xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE2]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x1x5714xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 34288] [1, 16, 2, 11427]
    // CHECK-SAME:      : tensor<1x16x2x80000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x2x11427xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE3]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x1x5714xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 45716] [1, 16, 2, 11427]
    // CHECK-SAME:      : tensor<1x16x2x80000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x2x11427xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE4]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x1x5714xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 57144] [1, 16, 2, 11427]
    // CHECK-SAME:      : tensor<1x16x2x80000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x2x11427xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE5]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x1x5714xf16, {order = #NHWC}>

    // CHECK:       [[INPUT_TILE6:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 68572] [1, 16, 2, 11427]
    // CHECK-SAME:      : tensor<1x16x2x80000xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x2x11427xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE6:%.+]] = VPU.NCE.MaxPool([[INPUT_TILE6]], [[WEIGHTS_TABLE]], [[ACTIVATION_WINDOW]]) {
    // CHECK-SAME:      pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}
    // CHECK-SAME:      } -> tensor<1x16x1x5714xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]], [[OUTPUT_TILE6]]) {
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 5715], [0, 0, 0, 11430], [0, 0, 0, 17144], [0, 0, 0, 22858], [0, 0, 0, 28572], [0, 0, 0, 34286]
    // CHECK-SAME:      -> tensor<1x16x1x40000xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x40000xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @NoTileWithSOH
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x100x100xf16, {order = #NHWC}>
func.func @NoTileWithSOH(
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

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<128x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<128x32x3x3xf16, {order = #NHWC}>
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
func.func @TileWithSOH(
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

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<32x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<32x16x3x3xf16, {order = #NHWC}>

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
func.func @NoTileWithSOK(
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

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<240x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS:%.+]] = const.Declare tensor<240x32x7x7xf16, {order = #NHWC}>
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
func.func @TileWithSOK(
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

    // CHECK-DAG:       [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS2:%.+]] = const.Declare tensor<256x32x7x7xf16, {order = #NHWC}>
    // CHECK-DAG:       [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS1:%.+]] = const.Declare tensor<256x32x7x7xf16, {order = #NHWC}>
    // CHECK-DAG:       [[WEIGHTS_TABLE0:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-DAG:       [[WEIGHTS0:%.+]] = const.Declare tensor<256x32x7x7xf16, {order = #NHWC}>

    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS0]], [[WEIGHTS_TABLE0]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64}
    // CHECK-SAME:          rawFilterShape = [256, 32, 7, 7]
    // CHECK-SAME:          tensor<1x256x30x30xf16, {order = #NHWC}>

    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS1]], [[WEIGHTS_TABLE1]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64}
    // CHECK-SAME:          rawFilterShape = [256, 32, 7, 7]
    // CHECK-SAME:          tensor<1x256x30x30xf16, {order = #NHWC}>

    // CHECK:       [[CONV3:%.+]] = VPU.NCE.Convolution([[INPUT]], [[WEIGHTS2]], [[WEIGHTS_TABLE2]])
    // CHECK-SAME:          multiClusterStrategy = "SplitOverKernel"
    // CHECK-SAME:          pad = {bottom = 3 : i64, left = 3 : i64, right = 3 : i64, top = 3 : i64}
    // CHECK-SAME:          rawFilterShape = [256, 32, 7, 7]
    // CHECK-SAME:          tensor<1x256x30x30xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV1]], [[CONV2]], [[CONV3]])

    // CHECK:       return [[CONCAT]] : tensor<1x768x30x30xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @LargeConstPipeliningSOKFor
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x256x14x14xf16, {order = #NHWC}>
func.func @LargeConstPipeliningSOKFor(
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

    // CHECK-DAG:       [[WEIGHTS_TABLE2:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-SAME:          [#const.SubView<[256, 0, 0, 0], [256, 1, 1, 4]>]
    // CHECK-DAG:       [[WEIGHTS2:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}>
    // CHECK-SAME:          [#const.Reorder<#NHWC>, #const.SubView<[256, 0, 0, 0], [256, 256, 3, 3]>]
    // CHECK-DAG:       [[WEIGHTS_TABLE1:%.+]] = const.Declare tensor<256x1x1x4xsi32>
    // CHECK-SAME:          [#const.SubView<[0, 0, 0, 0], [256, 1, 1, 4]>]
    // CHECK-DAG:       [[WEIGHTS1:%.+]] = const.Declare tensor<256x256x3x3xf16, {order = #NHWC}>
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

// CHECK-LABEL: func.func @SplitNCEEltwise
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x512x28x28xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x512x28x28xf16, {order = #NHWC}>
func.func @SplitNCEEltwise(
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
    // CHECK-SAME:      {op_type = "ADD"}
    // CHECK-SAME:      -> tensor<1x256x28x28xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:       [[INPUT_0_1:%.+]] = VPU.Slice [[INPUT_0]] [0, 256, 0, 0] [1, 256, 28, 28]
    // CHECK-SAME:      : tensor<1x512x28x28xf16, {order = #NHWC}> to tensor<1x256x28x28xf16, {order = #NHWC}>
    // CHECK:       [[INPUT_1_1:%.+]] = VPU.Slice [[INPUT_1]] [0, 256, 0, 0] [1, 256, 28, 28]
    // CHECK-SAME:      : tensor<1x512x28x28xf16, {order = #NHWC}> to tensor<1x256x28x28xf16, {order = #NHWC}>

    // CHECK:       [[ELTWISE_1:%.+]] = VPU.NCE.Eltwise([[INPUT_0_1]], [[INPUT_1_1]])
    // CHECK-SAME:      {op_type = "ADD"}
    // CHECK-SAME:      -> tensor<1x256x28x28xf16, {order = #NHWC}>

    // Concat
    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[ELTWISE_0]], [[ELTWISE_1]])
    // CHECK-SAME:      : tensor<1x256x28x28xf16, {order = #NHWC}>, tensor<1x256x28x28xf16, {order = #NHWC}>
    // CHECK-SAME:      -> tensor<1x512x28x28xf16, {order = #NHWC}>

    // return [[CONCAT]] : tensor<1x512x28x28xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @NoPrefetchingForEltwise
// CHECK-SAME:        [[INPUT_0:%arg[0-9]]]: tensor<1x32x70x70xf16, {order = #NHWC}>,
// CHECK-SAME:        [[INPUT_1:%arg[0-9]]]: tensor<1x64x70x70xf16, {order = #NHWC}>
func.func @NoPrefetchingForEltwise(
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

    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<64x1x1x4xsi32> = dense<1>
    // CHECK-DAG:       [[WEIGHTS:%.+]]       = const.Declare tensor<64x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>

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
func.func @SplitSparseNCEConvOverOH(%arg0: tensor<1x32x80x80xf16, {order = #NHWC}>) -> tensor<1x160x80x80xf16, {order = #NHWC}> {
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

    // CHECK-DAG:        [[WEIGHTS_TABLE_TILE:%.+]] = const.Declare tensor<160x1x1x4xsi32, {order = #NCHW}> = dense<10>

    // CHECK-DAG:        [[WEIGHTS_TILE:%.+]] = const.Declare tensor<160x1x1x384xi1> = dense<1.000000e+00>

    // CHECK-DAG:        [[WEIGHTS_SM_TILE:%.+]] = const.Declare tensor<160x32x3x3xf16, {order = #NHWC}> = dense<1.000000e+00>

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
func.func @SplitNCEAveragePoolOverW(%arg0: tensor<1x16x7x11520xf16, {order = #NHWC}>) -> tensor<1x16x1x11520xf16, {order = #NHWC}> {
    %0 = VPU.NCE.AveragePool(%arg0) {kernel_size = [7, 1], pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64}, ppe = {clamp_high = 2147483647 : i64, clamp_low = -2147483648 : i64, lrelu_mult = 1 : i64, lrelu_shift = 0 : i64, mode = "NOOP", quant_scale = [2.500000e-01]}, strides = [1, 1]} -> tensor<1x16x1x11520xf16, {order = #NHWC}>
    return %0 : tensor<1x16x1x11520xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 7, 2880]
    // CHECK-SAME:      tensor<1x16x7x11520xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x2880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE0]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x2880xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 2880] [1, 16, 7, 2880]
    // CHECK-SAME:      tensor<1x16x7x11520xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x2880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE1]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x2880xf16, {order = #NHWC}>

    // Tile 2

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 5760] [1, 16, 7, 2880]
    // CHECK-SAME:      tensor<1x16x7x11520xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x2880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE2]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x2880xf16, {order = #NHWC}>

    // Tile 3

    // CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 8640] [1, 16, 7, 2880]
    // CHECK-SAME:      tensor<1x16x7x11520xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x7x2880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.NCE.AveragePool([[INPUT_TILE3]]) {kernel_size = [7, 1]
    // CHECK-SAME:      -> tensor<1x16x1x2880xf16, {order = #NHWC}>


    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 2880], [0, 0, 0, 5760], [0, 0, 0, 8640]
    // CHECK-SAME:      -> tensor<1x16x1x11520xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x16x1x11520xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @SplitAveragePoolOverW
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x1x7x368640xf16, {order = #NHWC}>
func.func @SplitAveragePoolOverW(%arg0: tensor<1x1x7x368640xf16, {order = #NHWC}>) -> tensor<1x1x1x368640xf16, {order = #NHWC}> {
    %0 = VPU.AvgPool(%arg0) {exclude_pads, kernel_size = [7, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x1x7x368640xf16, {order = #NHWC}> -> tensor<1x1x1x368640xf16, {order = #NHWC}>

    return %0 : tensor<1x1x1x368640xf16, {order = #NHWC}>

    // Tile 0

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 1, 7, 122880]
    // CHECK-SAME:      : tensor<1x1x7x368640xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1x7x122880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.AvgPool([[INPUT_TILE0]])
    // CHECK-SAME:      -> tensor<1x1x1x122880xf16, {order = #NHWC}>

    // Tile 1

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 122880] [1, 1, 7, 122880]
    // CHECK-SAME:      : tensor<1x1x7x368640xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1x7x122880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.AvgPool([[INPUT_TILE1]])
    // CHECK-SAME:      -> tensor<1x1x1x122880xf16, {order = #NHWC}>

    // Tile 2

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 245760] [1, 1, 7, 122880]
    // CHECK-SAME:      : tensor<1x1x7x368640xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x1x7x122880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.AvgPool([[INPUT_TILE2]])
    // CHECK-SAME:      -> tensor<1x1x1x122880xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 0, 0, 122880], [0, 0, 0, 245760]
    // CHECK-SAME:      -> tensor<1x1x1x368640xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x1x368640xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @SplitOverWForSOHCompatibility
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x16x8x10001xf16, {order = #NHWC}>
func.func @SplitOverWForSOHCompatibility(%arg0: tensor<1x16x8x10001xf16, {order = #NHWC}>) -> tensor<1x16x8x10001xf16, {order = #NHWC}> {
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

    // CHECK-DAG:       [[ACT_WIN:%.+]] = const.Declare tensor<1x1x1x16xui8> = dense<1> : tensor<1x1x1x16xui8>
    // CHECK-DAG:       [[WEIGHTS_TABLE:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1> : tensor<16x1x1x4xsi32>

    // Tile 0
    // CHECK:       [[SLICE_0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 16, 8, 3334]
    // CHECK-SAME:      tensor<1x16x8x10001xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x8x3334xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_0:%.+]] = VPU.NCE.MaxPool([[SLICE_0]], [[WEIGHTS_TABLE]], [[ACT_WIN]])
    // CHECK-SAME:       multiClusterStrategy = "SplitOverHeight",
    // CHECK-SAME:       }
    // CHECK-SAME:      -> tensor<1x16x8x3334xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:       [[SLICE_1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 3334] [1, 16, 8, 3334]
    // CHECK-SAME:      tensor<1x16x8x10001xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x8x3334xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_1:%.+]] = VPU.NCE.MaxPool([[SLICE_1]], [[WEIGHTS_TABLE]], [[ACT_WIN]])
    // CHECK-SAME:       multiClusterStrategy = "SplitOverHeight",
    // CHECK-SAME:       }
    // CHECK-SAME:      -> tensor<1x16x8x3334xf16, {order = #NHWC}>

    // Tile 2
    // CHECK:       [[SLICE_2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 6668] [1, 16, 8, 3333]
    // CHECK-SAME:      tensor<1x16x8x10001xf16, {order = #NHWC}>
    // CHECK-SAME:      to tensor<1x16x8x3333xf16, {order = #NHWC}>
    // CHECK:       [[MAXPOOL_2:%.+]] = VPU.NCE.MaxPool([[SLICE_2]], [[WEIGHTS_TABLE]], [[ACT_WIN]])
    // CHECK-SAME:       multiClusterStrategy = "SplitOverHeight",
    // CHECK-SAME:       }
    // CHECK-SAME:      -> tensor<1x16x8x3333xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[MAXPOOL_0]], [[MAXPOOL_1]], [[MAXPOOL_2]])
    // CHECK:       return [[CONCAT]] : tensor<1x16x8x10001xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: func.func @PrefetchingNCEToUPATask
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x64x80x80xf16>
func.func @PrefetchingNCEToUPATask(%arg0: tensor<1x64x80x80xf16>) -> tensor<1x48x80x80xf16, {order = #NHWC}> {
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

    // CHECK-DAG:       [[WT_2:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1>
    // CHECK-DAG:       [[WEIGHTS_2:%.+]] = const.Declare tensor<16x64x5x5xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:       [[WT_1:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1>
    // CHECK-DAG:       [[WEIGHTS_1:%.+]] = const.Declare tensor<16x64x5x5xf16, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-DAG:       [[WT_0:%.+]] = const.Declare tensor<16x1x1x4xsi32> = dense<1>
    // CHECK-DAG:       [[WEIGHTS_0:%.+]] = const.Declare tensor<16x64x5x5xf16, {order = #NHWC}> = dense<1.000000e+00>

    // CHECK:       [[MEMPERMUTE:%.+]] = VPU.MemPermute([[INPUT]]) {dst_order = #NHWC, mem_perm = #NHWC}
    // CHECK-SAME:      : tensor<1x64x80x80xf16> -> tensor<1x64x80x80xf16, {order = #NHWC}>

    // Tile 0
    // CHECK:       [[CONV:%.+]] = VPU.NCE.Convolution([[MEMPERMUTE]], [[WEIGHTS_0]], [[WT_0]]) {
    // CHECK-SAME:      pad = {bottom = 2 : i64, left = 2 : i64, right = 2 : i64, top = 2 : i64},
    // CHECK-SAME:      rawFilterShape = [16, 64, 5, 5],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x80x80xf16, {order = #NHWC}>

    // Tile 1
    // CHECK:       [[CONV1:%.+]] = VPU.NCE.Convolution([[MEMPERMUTE]], [[WEIGHTS_1]], [[WT_1]]) {
    // CHECK-SAME:      pad = {bottom = 2 : i64, left = 2 : i64, right = 2 : i64, top = 2 : i64},
    // CHECK-SAME:      rawFilterShape = [16, 64, 5, 5],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x80x80xf16, {order = #NHWC}>

    // Tile 2
    // CHECK:       [[CONV2:%.+]] = VPU.NCE.Convolution([[MEMPERMUTE]], [[WEIGHTS_2]], [[WT_2]]) {
    // CHECK-SAME:      pad = {bottom = 2 : i64, left = 2 : i64, right = 2 : i64, top = 2 : i64},
    // CHECK-SAME:      rawFilterShape = [16, 64, 5, 5],
    // CHECK-SAME:      strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x16x80x80xf16, {order = #NHWC}>

    // CHECK:       [[CONCAT:%.+]] = VPU.Concat([[CONV]], [[CONV1]], [[CONV2]])
    // CHECK:       return [[CONCAT]]
}

// -----

// CHECK-LABEL: @ClampSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @ClampSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Clamp(%arg0) {max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Clamp([[INPUT_TILE0]]) {
// CHECK-SAME:    max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Clamp([[INPUT_TILE1]]) {
// CHECK-SAME:    max = 1.000000e+00 : f64, min = -1.000000e+00 : f64} : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @ReLUSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @ReLUSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.ReLU(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.ReLU([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.ReLU([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @LogSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @LogSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Log(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Log([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Log([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @AbsSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @AbsSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Abs(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Abs([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Abs([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

func.func @SplitFloorModEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.FloorMod(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitFloorModEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.FloorMod([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.FloorMod([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitPowerEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Power(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitPowerEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Power([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Power([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitLogicalOrEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.LogicalOr(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitLogicalOrEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LogicalOr([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LogicalOr([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitLogicalXorEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.LogicalXor(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitLogicalXorEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LogicalXor([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LogicalXor([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Equal(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Equal([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Equal([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitNotEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.NotEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitNotEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NotEqual([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NotEqual([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitSoftPlusActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.SoftPlus(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SplitSoftPlusActivationSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.SoftPlus([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.SoftPlus([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func.func @SplitLessEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Less(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitLessEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Less([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Less([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitLessEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.LessEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitLessEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LessEqual([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LessEqual([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitGreaterEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Greater(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitGreaterEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Greater([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Greater([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitGreaterEqualEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.GreaterEqual(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitGreaterEqualEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.GreaterEqual([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.GreaterEqual([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @SplitErfOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @SplitErfOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Erf(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Erf([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Erf([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @SplitFloorOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @SplitFloorOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Floor(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Floor([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Floor([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// CHECK-LABEL: @TanSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @TanSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Tan(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tan([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tan([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @SwishSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @SwishSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Swish(%arg0) {beta_value = 1.000000e+00 : f64} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Swish([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Swish([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

// CHECK-LABEL: @HSigmoidSplitOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
func.func @HSigmoidSplitOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.HSigmoid(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.HSigmoid([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.HSigmoid([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

func.func @SplitNegativeActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Negative(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Negative([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Negative([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func.func @SplitCeilingActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Ceiling(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>

    // CHECK-LABEL: @SplitCeilingActivationSw
    // CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Ceiling([[INPUT_TILE0]])
    // CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
    // CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Ceiling([[INPUT_TILE1]])
    // CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
    // CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

}

// -----

func.func @SplitSignActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Sign(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SplitSignActivationSw
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Sign([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Sign([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func.func @SplitSelectEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>, %arg2: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.Select(%arg0, %arg1, %arg2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
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
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x86x256xf16>, tensor<1x10x86x256xf16>, tensor<1x10x86x256xf16> -> tensor<1x10x86x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 86, 0] [1, 10, 85, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x85x256xf16>

// CHECK:       [[INPUT_TILE4:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 86, 0] [1, 10, 85, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x85x256xf16>

// CHECK:       [[INPUT_TILE5:%.+]] = VPU.Slice [[INPUT_2]] [0, 0, 86, 0] [1, 10, 85, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x85x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Select([[INPUT_TILE3]], [[INPUT_TILE4]], [[INPUT_TILE5]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x85x256xf16>, tensor<1x10x85x256xf16>, tensor<1x10x85x256xf16> -> tensor<1x10x85x256xf16>

// CHECK:       [[INPUT_TILE6:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 171, 0] [1, 10, 85, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x85x256xf16>

// CHECK:       [[INPUT_TILE7:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 171, 0] [1, 10, 85, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x85x256xf16>

// CHECK:       [[INPUT_TILE8:%.+]] = VPU.Slice [[INPUT_2]] [0, 0, 171, 0] [1, 10, 85, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x85x256xf16>

// CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.Select([[INPUT_TILE6]], [[INPUT_TILE7]], [[INPUT_TILE8]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x85x256xf16>, tensor<1x10x85x256xf16>, tensor<1x10x85x256xf16> -> tensor<1x10x85x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 86, 0], [0, 0, 171, 0]
// CHECK-SAME:  : tensor<1x10x86x256xf16>, tensor<1x10x85x256xf16>, tensor<1x10x85x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitAndEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.And(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitAndEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.And([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.And([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

func.func @SplitRoundActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
  %0 = VPU.Round(%arg0) {mode = #IE.round_mode<HALF_TO_EVEN>} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
  return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SplitRoundActivationSw
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Round([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Round([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>

// -----

func.func @SplitGeluActivationSw(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {
    %0 = VPU.Gelu(%arg0) : tensor<1x8x80x1280xf16> -> tensor<1x8x80x1280xf16>
    return %0 : tensor<1x8x80x1280xf16>
}

// CHECK-LABEL: @SplitGeluActivationSw
// CHECK-SAME:      [[INPUT:%.+]]: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x1280xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Gelu([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Gelu([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 640]
// CHECK-SAME:  : tensor<1x8x80x640xf16>, tensor<1x8x80x640xf16> -> tensor<1x8x80x1280xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x1280xf16>


// -----

func.func @SplitReduceSum(%arg0: tensor<1x12x368x480xf16>) -> tensor<1x1x368x480xf16> {
  %cst = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
  %0 = VPU.ReduceSum(%arg0, %cst) {keep_dims} : tensor<1x12x368x480xf16>, tensor<1xsi32> -> tensor<1x1x368x480xf16>
  return %0 : tensor<1x1x368x480xf16>

    // CHECK-LABEL: @SplitReduceSum
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x12x368x480xf16>) -> tensor<1x1x368x480xf16> {

    // CHECK-DAG:       [[CST:%.*]] = const.Declare tensor<1xsi32> = dense<1> : tensor<1xsi64>, [#const.ConvertElemType<si32>]
    // CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 12, 123, 480]
    // CHECK-SAME:   : tensor<1x12x368x480xf16> to tensor<1x12x123x480xf16>
    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.ReduceSum([[INPUT_TILE0]], [[CST]]) {
    // CHECK-SAME:  keep_dims} : tensor<1x12x123x480xf16>, tensor<1xsi32> -> tensor<1x1x123x480xf16>

    // CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 123, 0] [1, 12, 123, 480]
    // CHECK-SAME:   : tensor<1x12x368x480xf16> to tensor<1x12x123x480xf16>
    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.ReduceSum([[INPUT_TILE1]], [[CST]]) {
    // CHECK-SAME:  keep_dims} : tensor<1x12x123x480xf16>, tensor<1xsi32> -> tensor<1x1x123x480xf16>

    // CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT]] [0, 0, 246, 0] [1, 12, 122, 480]
    // CHECK-SAME    : tensor<1x12x368x480xf16> to tensor<1x12x122x480xf16>
    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.ReduceSum([[INPUT_TILE2]], [[CST]]) {
    // CHECK-SAME:  keep_dims} : tensor<1x12x122x480xf16>, tensor<1xsi32> -> tensor<1x1x122x480xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 123, 0], [0, 0, 246, 0]
    // CHECK-SAME:   : tensor<1x1x123x480xf16>, tensor<1x1x123x480xf16>, tensor<1x1x122x480xf16> -> tensor<1x1x368x480xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x1x368x480xf16>
}

// -----

func.func @SplitTileOnTwoDim(%arg0: tensor<1x1x2880x50xf16>) -> tensor<1x2x8640x150xf16> {
  %0 = VPU.Tile(%arg0) {repeats_values = [1, 2, 3, 3]} : tensor<1x1x2880x50xf16> -> tensor<1x2x8640x150xf16>
  return %0 : tensor<1x2x8640x150xf16>

    // CHECK-LABEL: @SplitTileOnTwoDim
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x1x2880x50xf16>) -> tensor<1x2x8640x150xf16> {

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 2, 1, 1]} : tensor<1x1x2880x50xf16> -> tensor<1x2x2880x50xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 2, 1, 1]} : tensor<1x1x2880x50xf16> -> tensor<1x2x2880x50xf16>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 2, 1, 1]} : tensor<1x1x2880x50xf16> -> tensor<1x2x2880x50xf16>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 2, 1, 1]} : tensor<1x1x2880x50xf16> -> tensor<1x2x2880x50xf16>

    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 2, 1, 1]} : tensor<1x1x2880x50xf16> -> tensor<1x2x2880x50xf16>

    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 2, 1, 1]} : tensor<1x1x2880x50xf16> -> tensor<1x2x2880x50xf16>

    // CHECK:       [[OUTPUT_TILE6:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 2, 1, 1]} : tensor<1x1x2880x50xf16> -> tensor<1x2x2880x50xf16>

    // CHECK:       [[OUTPUT_TILE7:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 2, 1, 1]} : tensor<1x1x2880x50xf16> -> tensor<1x2x2880x50xf16>

    // CHECK:       [[OUTPUT_TILE8:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 2, 1, 1]} : tensor<1x1x2880x50xf16> -> tensor<1x2x2880x50xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]], [[OUTPUT_TILE6]], [[OUTPUT_TILE7]], [[OUTPUT_TILE8]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 50], [0, 0, 0, 100], [0, 0, 2880, 0], [0, 0, 2880, 50], [0, 0, 2880, 100], [0, 0, 5760, 0], [0, 0, 5760, 50], [0, 0, 5760, 100]
    // CHECK-SAME:   : tensor<1x2x2880x50xf16>, tensor<1x2x2880x50xf16>, tensor<1x2x2880x50xf16>, tensor<1x2x2880x50xf16>, tensor<1x2x2880x50xf16>, tensor<1x2x2880x50xf16>, tensor<1x2x2880x50xf16>, tensor<1x2x2880x50xf16>, tensor<1x2x2880x50xf16> -> tensor<1x2x8640x150xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x2x8640x150xf16>
}

// -----

func.func @SplitTileRepeatsNotOne(%arg0: tensor<3x2x723x25xf16>) -> tensor<6x4x2169x75xf16> {
  %0 = VPU.Tile(%arg0) {repeats_values = [2, 2, 3, 3]} : tensor<3x2x723x25xf16> -> tensor<6x4x2169x75xf16>
  return %0 : tensor<6x4x2169x75xf16>

    // CHECK-LABEL: @SplitTileRepeatsNotOne
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<3x2x723x25xf16>) -> tensor<6x4x2169x75xf16> {

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [2, 2, 1, 1]} : tensor<3x2x723x25xf16> -> tensor<6x4x723x25xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [2, 2, 1, 1]} : tensor<3x2x723x25xf16> -> tensor<6x4x723x25xf16>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [2, 2, 1, 1]} : tensor<3x2x723x25xf16> -> tensor<6x4x723x25xf16>

    // CHECK:       [[OUTPUT_TILE3:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [2, 2, 1, 1]} : tensor<3x2x723x25xf16> -> tensor<6x4x723x25xf16>

    // CHECK:       [[OUTPUT_TILE4:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [2, 2, 1, 1]} : tensor<3x2x723x25xf16> -> tensor<6x4x723x25xf16>

    // CHECK:       [[OUTPUT_TILE5:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [2, 2, 1, 1]} : tensor<3x2x723x25xf16> -> tensor<6x4x723x25xf16>

    // CHECK:       [[OUTPUT_TILE6:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [2, 2, 1, 1]} : tensor<3x2x723x25xf16> -> tensor<6x4x723x25xf16>

    // CHECK:       [[OUTPUT_TILE7:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [2, 2, 1, 1]} : tensor<3x2x723x25xf16> -> tensor<6x4x723x25xf16>

    // CHECK:       [[OUTPUT_TILE8:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [2, 2, 1, 1]} : tensor<3x2x723x25xf16> -> tensor<6x4x723x25xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]], [[OUTPUT_TILE3]], [[OUTPUT_TILE4]], [[OUTPUT_TILE5]], [[OUTPUT_TILE6]], [[OUTPUT_TILE7]], [[OUTPUT_TILE8]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 25], [0, 0, 0, 50], [0, 0, 723, 0], [0, 0, 723, 25], [0, 0, 723, 50], [0, 0, 1446, 0], [0, 0, 1446, 25], [0, 0, 1446, 50]
    // CHECK-SAME:   : tensor<6x4x723x25xf16>, tensor<6x4x723x25xf16>, tensor<6x4x723x25xf16>, tensor<6x4x723x25xf16>, tensor<6x4x723x25xf16>, tensor<6x4x723x25xf16>, tensor<6x4x723x25xf16>, tensor<6x4x723x25xf16>, tensor<6x4x723x25xf16> -> tensor<6x4x2169x75xf16>

    // CHECK:       return [[OUTPUT]] : tensor<6x4x2169x75xf16>
}

// -----

func.func @SplitTileRepeatsIsOne(%arg0: tensor<2x3x360x50xf16>) -> tensor<6x3x1080x100xf16> {
  %0 = VPU.Tile(%arg0) {repeats_values = [3, 1, 3, 2]} : tensor<2x3x360x50xf16> -> tensor<6x3x1080x100xf16>
  return %0 : tensor<6x3x1080x100xf16>

    // CHECK-LABEL: @SplitTileRepeatsIsOne
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<2x3x360x50xf16>) -> tensor<6x3x1080x100xf16> {

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [3, 1, 1, 2]} : tensor<2x3x360x50xf16> -> tensor<6x3x360x100xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [3, 1, 1, 2]} : tensor<2x3x360x50xf16> -> tensor<6x3x360x100xf16>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [3, 1, 1, 2]} : tensor<2x3x360x50xf16> -> tensor<6x3x360x100xf16>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 360, 0], [0, 0, 720, 0]
    // CHECK-SAME:   : tensor<6x3x360x100xf16>, tensor<6x3x360x100xf16>, tensor<6x3x360x100xf16> -> tensor<6x3x1080x100xf16>

    // CHECK:       return [[OUTPUT]] : tensor<6x3x1080x100xf16>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

func.func @SplitTileModelCase(%arg0: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x65536xf16, {order = #NHWC}> {
  %0 = VPU.Tile(%arg0) {repeats_values = [1, 1, 1, 65536]} : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x65536xf16, {order = #NHWC}>
  return %0 : tensor<1x32x1x65536xf16, {order = #NHWC}>

    // CHECK-LABEL: @SplitTileModelCase
    // CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x32x1x1xf16, {order = #NHWC}>) -> tensor<1x32x1x65536xf16, {order = #NHWC}> {

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 1, 21846]}
    // CHECK-SAME:   : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x21846xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 1, 21845]}
    // CHECK-SAME:   : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x21845xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE2:%.+]] = VPU.Tile([[INPUT]]) {
    // CHECK-SAME:  repeats_values = [1, 1, 1, 21845]}
    // CHECK-SAME:   : tensor<1x32x1x1xf16, {order = #NHWC}> -> tensor<1x32x1x21845xf16, {order = #NHWC}>

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]])
    // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 21846], [0, 0, 0, 43691]
    // CHECK-SAME:   : tensor<1x32x1x21846xf16, {order = #NHWC}>, tensor<1x32x1x21845xf16, {order = #NHWC}>, tensor<1x32x1x21845xf16, {order = #NHWC}> -> tensor<1x32x1x65536xf16, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x32x1x65536xf16, {order = #NHWC}>
}

// -----

func.func @SplitTopK(%arg0: tensor<1x5x512x512xf16>) -> (tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>) {
    %cst = const.Declare tensor<si32> = dense<1> : tensor<si64>, [#const.ConvertElemType<si32>]
    %output_values, %target_shape = VPU.TopK(%arg0, %cst) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x512x512xf16>, tensor<si32> -> tensor<1x1x512x512xf16>, tensor<1x1x512x512xsi32>
    %0 = VPU.Convert(%output_values) {dstElemType = f32} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf32>
    return %0, %target_shape : tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>

    // CHECK-LABEL: @SplitTopK
    // CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x5x512x512xf16>) -> (tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>) {
    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<si32> = dense<1> : tensor<si64>, [#const.ConvertElemType<si32>]
    // CHECK: [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 5, 171, 512] : tensor<1x5x512x512xf16> to tensor<1x5x171x512xf16>
    // CHECK: [[OUTPUT_TILE0:%.+]], [[TARGET_TILE0:%.+]] = VPU.TopK([[INPUT_TILE0]], [[CST]]) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x171x512xf16>, tensor<si32> -> tensor<1x1x171x512xf16>, tensor<1x1x171x512xsi32>
    // CHECK: [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 171, 0] [1, 5, 171, 512] : tensor<1x5x512x512xf16> to tensor<1x5x171x512xf16>
    // CHECK: [[OUTPUT_TILE1:%.+]], [[TARGET_TILE1:%.+]] = VPU.TopK([[INPUT_TILE1]], [[CST]]) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x171x512xf16>, tensor<si32> -> tensor<1x1x171x512xf16>, tensor<1x1x171x512xsi32>
    // CHECK: [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 342, 0] [1, 5, 170, 512] : tensor<1x5x512x512xf16> to tensor<1x5x170x512xf16>
    // CHECK: [[OUTPUT_TILE2:%.+]], [[TARGET_TILE2:%.+]] = VPU.TopK([[INPUT_TILE2]], [[CST]]) {axis = 1 : i64, element_type = si32, mode = #IE.topk_mode<MAX>, sort = #IE.topk_sort_type<SORT_INDICES>} : tensor<1x5x170x512xf16>, tensor<si32> -> tensor<1x1x170x512xf16>, tensor<1x1x170x512xsi32>
    // CHECK: [[OUTPUT_VALUE:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]], [[OUTPUT_TILE2]]) {static_offsets =
    // CHECK: [[TARGET_SHAPE:%.+]] = VPU.Concat([[TARGET_TILE0]], [[TARGET_TILE1]], [[TARGET_TILE2]]) {static_offsets =
    // CHECK: [[OUTPUT_VALUE_CONV:%.+]] = VPU.Convert([[OUTPUT_VALUE]]) {dstElemType = f32} : tensor<1x1x512x512xf16> -> tensor<1x1x512x512xf32>
    // CHECK: return [[OUTPUT_VALUE_CONV]], [[TARGET_SHAPE]] : tensor<1x1x512x512xf32>, tensor<1x1x512x512xsi32>
}

// -----

// CHECK-LABEL: func.func @SplitStridedSliceOverW
// CHECK-SAME:  [[INPUT:%arg[0-9]]]: tensor<1x8x80x1280xf16>
func.func @SplitStridedSliceOverW(%arg0: tensor<1x8x80x1280xf16>) -> tensor<1x8x80x640xf16> {
    %0 = VPU.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [1, 8, 80, 1280], new_axis_mask = [0, 0, 0, 0], shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<1x8x80x1280xf16> -> tensor<1x8x80x640xf16>
    return %0 : tensor<1x8x80x640xf16>
  }

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.StridedSlice([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x320xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 640] [1, 8, 80, 640]
// CHECK-SAME:  : tensor<1x8x80x1280xf16> to tensor<1x8x80x640xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.StridedSlice([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x8x80x640xf16> -> tensor<1x8x80x320xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 0, 320]
// CHECK-SAME:  : tensor<1x8x80x320xf16>, tensor<1x8x80x320xf16> -> tensor<1x8x80x640xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x8x80x640xf16>


// -----

func.func @SplitLogicalNotEltwiseSw(%arg0: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.LogicalNot(%arg0) : tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitLogicalNotEltwiseSw
// CHECK-SAME:      [[INPUT:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>
// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.LogicalNot([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>
// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.LogicalNot([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:   @SplitNCEConvOverOCInputOversize
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x32x64xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOCInputOversize(%arg0: tensor<1x512x32x64xf16, {order = #NHWC}>) -> tensor<1x256x32x64xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
  %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
      pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
      rawFilterShape = [256, 512, 3, 3],
      strides = [1, 1]
  } -> tensor<1x256x32x64xf16, {order = #NHWC}>

  return %0 : tensor<1x256x32x64xf16, {order = #NHWC}>

  // CHECK-DAG:       %cst = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[192, 0, 0, 0], [64, 1, 1, 4]>]
  // CHECK-DAG:       %cst_0 = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[192, 0, 0, 0], [64, 512, 3, 3]>]
  // CHECK-DAG:       %cst_1 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[128, 0, 0, 0], [64, 1, 1, 4]>]
  // CHECK-DAG:       %cst_2 = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[128, 0, 0, 0], [64, 512, 3, 3]>]
  // CHECK-DAG:       %cst_3 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[64, 0, 0, 0], [64, 1, 1, 4]>]
  // CHECK-DAG:       %cst_4 = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[64, 0, 0, 0], [64, 512, 3, 3]>]
  // CHECK-DAG:       %cst_5 = const.Declare tensor<64x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [64, 1, 1, 4]>]
  // CHECK-DAG:       %cst_6 = const.Declare tensor<64x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [64, 512, 3, 3]>]
  // CHECK:       %0 = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %1 = VPU.NCE.Convolution(%0, %cst_6, %cst_5) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %2 = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %3 = VPU.NCE.Convolution(%2, %cst_6, %cst_5) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %4 = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %5 = VPU.NCE.Convolution(%4, %cst_4, %cst_3) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %6 = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %7 = VPU.NCE.Convolution(%6, %cst_4, %cst_3) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %8 = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %9 = VPU.NCE.Convolution(%8, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %10 = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %11 = VPU.NCE.Convolution(%10, %cst_2, %cst_1) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %12 = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %13 = VPU.NCE.Convolution(%12, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %14 = VPU.Slice [[INPUT]] [0, 0, 15, 0] [1, 512, 17, 64] : tensor<1x512x32x64xf16, {order = #NHWC}> to tensor<1x512x17x64xf16, {order = #NHWC}>
  // CHECK:       %15 = VPU.NCE.Convolution(%14, %cst_0, %cst) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [64, 512, 3, 3], strides = [1, 1]} -> tensor<1x64x16x64xf16, {order = #NHWC}>
  // CHECK:       %16 = VPU.Concat(%1, %3, %5, %7, %9, %11, %13, %15)
  // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 16, 0], [0, 64, 0, 0], [0, 64, 16, 0], [0, 128, 0, 0], [0, 128, 16, 0], [0, 192, 0, 0], [0, 192, 16, 0]
  // CHECK-SAME:  : tensor<1x64x16x64xf16, {order = #NHWC}>, tensor<1x64x16x64xf16, {order = #NHWC}>, tensor<1x64x16x64xf16, {order = #NHWC}>, tensor<1x64x16x64xf16, {order = #NHWC}>, tensor<1x64x16x64xf16, {order = #NHWC}>, tensor<1x64x16x64xf16, {order = #NHWC}>, tensor<1x64x16x64xf16, {order = #NHWC}>, tensor<1x64x16x64xf16, {order = #NHWC}> -> tensor<1x256x32x64xf16, {order = #NHWC}>
  // CHECK:       return %16 : tensor<1x256x32x64xf16, {order = #NHWC}>
}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>
// CHECK-LABEL:   @SplitNCEConvOverOHFilterOversize
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x512x64x64xf16, {order = #NHWC}>
func.func @SplitNCEConvOverOHFilterOversize(%arg0: tensor<1x512x64x64xf16, {order = #NHWC}>) -> tensor<1x256x64x64xf16, {order = #NHWC}> {
  %weights = const.Declare tensor<256x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>]
  %weights_table = const.Declare tensor<256x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>

  %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
      pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64},
      rawFilterShape = [256, 512, 3, 3],
      strides = [1, 1]
  } -> tensor<1x256x64x64xf16, {order = #NHWC}>

  return %0 : tensor<1x256x64x64xf16, {order = #NHWC}>

  // CHECK-DAG:       %cst = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[128, 0, 0, 0], [128, 1, 1, 4]>]
  // CHECK-DAG:       %cst_0 = const.Declare tensor<128x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[128, 0, 0, 0], [128, 512, 3, 3]>]
  // CHECK-DAG:       %cst_1 = const.Declare tensor<128x1x1x4xsi32> = dense<1> : tensor<256x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [128, 1, 1, 4]>]
  // CHECK-DAG:       %cst_2 = const.Declare tensor<128x512x3x3xf16, {order = #NHWC}> = dense<1.000000e+00> : tensor<256x512x3x3xf16>, [#const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [128, 512, 3, 3]>]
  // CHECK:       %0 = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %1 = VPU.NCE.Convolution(%0, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x8x64xf16, {order = #NHWC}>
  // CHECK:       %2 = VPU.Slice [[INPUT]] [0, 0, 7, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %3 = VPU.NCE.Convolution(%2, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %4 = VPU.Slice [[INPUT]] [0, 0, 14, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %5 = VPU.NCE.Convolution(%4, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %6 = VPU.Slice [[INPUT]] [0, 0, 21, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %7 = VPU.NCE.Convolution(%6, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %8 = VPU.Slice [[INPUT]] [0, 0, 28, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %9 = VPU.NCE.Convolution(%8, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %10 = VPU.Slice [[INPUT]] [0, 0, 35, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %11 = VPU.NCE.Convolution(%10, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %12 = VPU.Slice [[INPUT]] [0, 0, 42, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %13 = VPU.NCE.Convolution(%12, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %14 = VPU.Slice [[INPUT]] [0, 0, 49, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %15 = VPU.NCE.Convolution(%14, %cst_2, %cst_1) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %16 = VPU.Slice [[INPUT]] [0, 0, 56, 0] [1, 512, 8, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x8x64xf16, {order = #NHWC}>
  // CHECK:       %17 = VPU.NCE.Convolution(%16, %cst_2, %cst_1) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %18 = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %19 = VPU.NCE.Convolution(%18, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 1 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x8x64xf16, {order = #NHWC}>
  // CHECK:       %20 = VPU.Slice [[INPUT]] [0, 0, 7, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %21 = VPU.NCE.Convolution(%20, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %22 = VPU.Slice [[INPUT]] [0, 0, 14, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %23 = VPU.NCE.Convolution(%22, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %24 = VPU.Slice [[INPUT]] [0, 0, 21, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %25 = VPU.NCE.Convolution(%24, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %26 = VPU.Slice [[INPUT]] [0, 0, 28, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %27 = VPU.NCE.Convolution(%26, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %28 = VPU.Slice [[INPUT]] [0, 0, 35, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %29 = VPU.NCE.Convolution(%28, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %30 = VPU.Slice [[INPUT]] [0, 0, 42, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %31 = VPU.NCE.Convolution(%30, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %32 = VPU.Slice [[INPUT]] [0, 0, 49, 0] [1, 512, 9, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x9x64xf16, {order = #NHWC}>
  // CHECK:       %33 = VPU.NCE.Convolution(%32, %cst_0, %cst) {pad = {bottom = 0 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %34 = VPU.Slice [[INPUT]] [0, 0, 56, 0] [1, 512, 8, 64] : tensor<1x512x64x64xf16, {order = #NHWC}> to tensor<1x512x8x64xf16, {order = #NHWC}>
  // CHECK:       %35 = VPU.NCE.Convolution(%34, %cst_0, %cst) {pad = {bottom = 1 : i64, left = 1 : i64, right = 1 : i64, top = 0 : i64}, rawFilterShape = [128, 512, 3, 3], strides = [1, 1]} -> tensor<1x128x7x64xf16, {order = #NHWC}>
  // CHECK:       %36 = VPU.Concat(%1, %3, %5, %7, %9, %11, %13, %15, %17, %19, %21, %23, %25, %27, %29, %31, %33, %35)
  // CHECK-SAME:  [0, 0, 0, 0], [0, 0, 8, 0], [0, 0, 15, 0], [0, 0, 22, 0], [0, 0, 29, 0], [0, 0, 36, 0], [0, 0, 43, 0], [0, 0, 50, 0], [0, 0, 57, 0], [0, 128, 0, 0], [0, 128, 8, 0], [0, 128, 15, 0], [0, 128, 22, 0], [0, 128, 29, 0], [0, 128, 36, 0], [0, 128, 43, 0], [0, 128, 50, 0], [0, 128, 57, 0]
  // CHECK-SAME:  : tensor<1x128x8x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x8x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}>, tensor<1x128x7x64xf16, {order = #NHWC}> -> tensor<1x256x64x64xf16, {order = #NHWC}>
  // CHECK:       return %36 : tensor<1x256x64x64xf16, {order = #NHWC}>
}

// -----

#NCHW = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

!qElemType0 = !quant.uniform<u8:f16, 0.96372549019607844>
!qElemType1 = !quant.uniform<u8:f16, 0.054779411764705882>
!qElemType2 = !quant.uniform<u8<0:254>:f16, 8.7179349163385824E-4:127>

// CHECK-LABEL:   @SplitQuantNCEConvOverOCKeepInputShape
// CHECK-SAME:          [[INPUT:%arg[0-9]]]: tensor<1x1024x14x14x!qElemType0, {order = #NHWC}>
func.func  @SplitQuantNCEConvOverOCKeepInputShape(%arg0: tensor<1x1024x14x14x!qElemType0, {order = #NHWC}>) -> tensor<1x1024x7x7x!qElemType1, {order = #NHWC}> {
    %weights = const.Declare tensor<1024x1024x1x1x!qElemType2, {order = #NHWC}> = dense<1.000000e+00> : tensor<1024x1024x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>]
    %weights_table = const.Declare tensor<1024x1x1x4xsi32, {order = #NCHW}> = dense<10> : tensor<1024x1x1x4xsi32>

    %0 = VPU.NCE.Convolution(%arg0, %weights, %weights_table) {
        pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
        rawFilterShape = [1024, 1024, 1, 1],
        strides = [2, 2]
    } -> tensor<1x1024x7x7x!qElemType1, {order = #NHWC}>

    return %0 : tensor<1x1024x7x7x!qElemType1, {order = #NHWC}>

    // CHECK-DAG:        [[WEIGHTS_TABLE_TILE1:%.+]] = const.Declare tensor<512x1x1x4xsi32> = dense<10>
    // CHECK-SAME:      : tensor<1024x1x1x4xsi32>, [#const.SubView<[512, 0, 0, 0], [512, 1, 1, 4]>]

    // CHECK-DAG:        [[FILTER_TILE1:%.+]] = const.Declare tensor<512x1024x1x1x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1024x1024x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>, #const.SubView<[512, 0, 0, 0], [512, 1024, 1, 1]>]

    // CHECK-DAG:        [[WEIGHTS_TABLE_TILE0:%.+]] = const.Declare tensor<512x1x1x4xsi32> = dense<10>
    // CHECK-SAME:      : tensor<1024x1x1x4xsi32>, [#const.SubView<[0, 0, 0, 0], [512, 1, 1, 4]>]

    // CHECK-DAG:        [[FILTER_TILE0:%.+]] = const.Declare tensor<512x1024x1x1x!qElemType2, {order = #NHWC}> = dense<1.000000e+00>
    // CHECK-SAME:      : tensor<1024x1024x1x1xf16>, [#const.ConvertElemType<ui8>, #const.QuantCast<!qElemType2>, #const.Reorder<#NHWC>, #const.SubView<[0, 0, 0, 0], [512, 1024, 1, 1]>]

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE0]], [[WEIGHTS_TABLE_TILE0]])
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [512, 1024, 1, 1],
    // CHECK-SAME:          strides = [2, 2]
    // CHECK-SAME:          -> tensor<1x512x7x7x!qElemType1, {order = #NHWC}>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.NCE.Convolution([[INPUT]], [[FILTER_TILE1]], [[WEIGHTS_TABLE_TILE1]])
    // CHECK-SAME:          pad = {bottom = 0 : i64, left = 0 : i64, right = 0 : i64, top = 0 : i64},
    // CHECK-SAME:          rawFilterShape = [512, 1024, 1, 1],
    // CHECK-SAME:          strides = [2, 2]
    // CHECK-SAME:          -> tensor<1x512x7x7x!qElemType1, {order = #NHWC}>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:          [0, 0, 0, 0], [0, 512, 0, 0]
    // CHECK-SAME:          -> tensor<1x1024x7x7x!qElemType1, {order = #NHWC}>

    // CHECK:       return [[OUTPUT]] : tensor<1x1024x7x7x!qElemType1, {order = #NHWC}>
}

// -----

func.func @SplitSquaredDiffEltwiseSw(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {
  %0 = VPU.SquaredDiff(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x10x256x256xf16> -> tensor<1x10x256x256xf16>
  return %0 : tensor<1x10x256x256xf16>
}

// CHECK-LABEL: @SplitSquaredDiffEltwiseSw
// CHECK-SAME:      [[INPUT_0:%arg[0-9]]]: tensor<1x10x256x256xf16>, [[INPUT_1:%arg[0-9]]]: tensor<1x10x256x256xf16>) -> tensor<1x10x256x256xf16> {

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 0, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.SquaredDiff([[INPUT_TILE0]], [[INPUT_TILE1]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE2:%.+]] = VPU.Slice [[INPUT_0]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[INPUT_TILE3:%.+]] = VPU.Slice [[INPUT_1]] [0, 0, 128, 0] [1, 10, 128, 256]
// CHECK-SAME:   : tensor<1x10x256x256xf16> to tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.SquaredDiff([[INPUT_TILE2]], [[INPUT_TILE3]]) {
// CHECK-SAME:  auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x128x256xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 128, 0]
// CHECK-SAME:  : tensor<1x10x128x256xf16>, tensor<1x10x128x256xf16> -> tensor<1x10x256x256xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x10x256x256xf16>

// -----

// CHECK-LABEL: @ReLUSplitOverC
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x80x80x80xf16>) -> tensor<1x80x80x80xf16> {
func.func @ReLUSplitOverC(%arg0: tensor<1x80x80x80xf16>) -> tensor<1x80x80x80xf16> {
  %0 = VPU.ReLU(%arg0) : tensor<1x80x80x80xf16> -> tensor<1x80x80x80xf16>
  return %0 : tensor<1x80x80x80xf16>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 40, 80, 80]
// CHECK-SAME:  : tensor<1x80x80x80xf16> to tensor<1x40x80x80xf16>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.ReLU([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x40x80x80xf16> -> tensor<1x40x80x80xf16>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 40, 0, 0] [1, 40, 80, 80]
// CHECK-SAME:  : tensor<1x80x80x80xf16> to tensor<1x40x80x80xf16>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.ReLU([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x40x80x80xf16> -> tensor<1x40x80x80xf16>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 40, 0, 0]
// CHECK-SAME:  : tensor<1x40x80x80xf16>, tensor<1x40x80x80xf16> -> tensor<1x80x80x80xf16>

// CHECK:       return [[OUTPUT]] : tensor<1x80x80x80xf16>

}

// -----

#NHWC = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3, d1)>

// CHECK-LABEL: @ReLUSplitOverH
// CHECK-SAME:  [[INPUT:%.+]]: tensor<1x80x80x80xf16, {order = #NHWC}>) -> tensor<1x80x80x80xf16, {order = #NHWC}> {
func.func @ReLUSplitOverH(%arg0: tensor<1x80x80x80xf16, {order = #NHWC}>) -> tensor<1x80x80x80xf16, {order = #NHWC}> {
  %0 = VPU.ReLU(%arg0) : tensor<1x80x80x80xf16, {order = #NHWC}> -> tensor<1x80x80x80xf16, {order = #NHWC}>
  return %0 : tensor<1x80x80x80xf16, {order = #NHWC}>

// CHECK:       [[INPUT_TILE0:%.+]] = VPU.Slice [[INPUT]] [0, 0, 0, 0] [1, 80, 40, 80]
// CHECK-SAME:  : tensor<1x80x80x80xf16, {order = #NHWC}> to tensor<1x80x40x80xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.ReLU([[INPUT_TILE0]])
// CHECK-SAME:  : tensor<1x80x40x80xf16, {order = #NHWC}> -> tensor<1x80x40x80xf16, {order = #NHWC}>

// CHECK:       [[INPUT_TILE1:%.+]] = VPU.Slice [[INPUT]] [0, 0, 40, 0] [1, 80, 40, 80]
// CHECK-SAME:  : tensor<1x80x80x80xf16, {order = #NHWC}> to tensor<1x80x40x80xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.ReLU([[INPUT_TILE1]])
// CHECK-SAME:  : tensor<1x80x40x80xf16, {order = #NHWC}> -> tensor<1x80x40x80xf16, {order = #NHWC}>

// CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
// CHECK-SAME:  [0, 0, 0, 0], [0, 0, 40, 0]
// CHECK-SAME:  : tensor<1x80x40x80xf16, {order = #NHWC}>, tensor<1x80x40x80xf16, {order = #NHWC}> -> tensor<1x80x80x80xf16, {order = #NHWC}>

// CHECK:       return [[OUTPUT]] : tensor<1x80x80x80xf16, {order = #NHWC}>

}

// -----

// CHECK-LABEL: func.func @NoBiasConv
// CHECK-SAME:        [[INPUT:%arg[0-9]]]: tensor<1x32x64x64xf16>,
// CHECK-SAME:        [[FILTER:%arg[0-9]]]: tensor<256x32x3x3xf16>
func.func @NoBiasConv(
        %input: tensor<1x32x64x64xf16>,
        %filter: tensor<256x32x3x3xf16>)
          -> tensor<1x256x62x62xf16> {
    %1 = VPU.Convolution(%input, %filter) {
        dilations = [1, 1],
        pads_begin = [0, 0],
        pads_end = [0, 0],
        strides = [1, 1]
    } : tensor<1x32x64x64xf16>, tensor<256x32x3x3xf16> -> tensor<1x256x62x62xf16>
    return %1 : tensor<1x256x62x62xf16>

    // Tile 0

    // CHECK:       [[FILTER_TILE0:%.+]] = VPU.Slice [[FILTER]] [0, 0, 0, 0] [128, 32, 3, 3]
    // CHECK-SAME:      : tensor<256x32x3x3xf16> to tensor<128x32x3x3xf16>

    // CHECK:       [[OUTPUT_TILE0:%.+]] = VPU.Convolution([[INPUT]], [[FILTER_TILE0]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [0, 0]
    // CHECK-SAME:          pads_end = [0, 0]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x62x62xf16>

    // Tile 1

    // CHECK:       [[FILTER_TILE1:%.+]] = VPU.Slice [[FILTER]] [128, 0, 0, 0] [128, 32, 3, 3]
    // CHECK-SAME:      : tensor<256x32x3x3xf16> to tensor<128x32x3x3xf16>

    // CHECK:       [[OUTPUT_TILE1:%.+]] = VPU.Convolution([[INPUT]], [[FILTER_TILE1]])
    // CHECK-SAME:          dilations = [1, 1]
    // CHECK-SAME:          pads_begin = [0, 0]
    // CHECK-SAME:          pads_end = [0, 0]
    // CHECK-SAME:          strides = [1, 1]
    // CHECK-SAME:      -> tensor<1x128x62x62xf16>

    // Concat

    // CHECK:       [[OUTPUT:%.+]] = VPU.Concat([[OUTPUT_TILE0]], [[OUTPUT_TILE1]])
    // CHECK-SAME:      [0, 0, 0, 0], [0, 128, 0, 0]
    // CHECK-SAME:      -> tensor<1x256x62x62xf16>

    // CHECK:       return [[OUTPUT]] : tensor<1x256x62x62xf16>
}

// -----

// CHECK-LABEL: @DFT
// CHECK-SAME:   (%arg0: tensor<1x120x64x64x2xf16>) -> tensor<1x120x64x64x2xf16>
func.func @DFT(%arg0: tensor<1x120x64x64x2xf16>) -> tensor<1x120x64x64x2xf16> {
    %0 = VPU.DFT(%arg0) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x120x64x64x2xf16> -> tensor<1x120x64x64x2xf16>
    return %0 : tensor<1x120x64x64x2xf16>
    // CHECK: %0 = VPU.Slice %arg0 [0, 0, 0, 0, 0] [1, 60, 64, 64, 2] : tensor<1x120x64x64x2xf16> to tensor<1x60x64x64x2xf16>
    // CHECK: %1 = VPU.DFT(%0) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x60x64x64x2xf16> -> tensor<1x60x64x64x2xf16>
    // CHECK: %2 = VPU.Slice %arg0 [0, 60, 0, 0, 0] [1, 60, 64, 64, 2] : tensor<1x120x64x64x2xf16> to tensor<1x60x64x64x2xf16>
    // CHECK: %3 = VPU.DFT(%2) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x60x64x64x2xf16> -> tensor<1x60x64x64x2xf16>
    // CHECK{LITERAL}: %4 = VPU.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0, 0], [0, 60, 0, 0, 0]]} : tensor<1x60x64x64x2xf16>, tensor<1x60x64x64x2xf16> -> tensor<1x120x64x64x2xf16>
    // CHECK: return %4 : tensor<1x120x64x64x2xf16>
}

// CHECK-LABEL: @IDFT
// CHECK-SAME:   (%arg0: tensor<1x120x64x64x2xf16>) -> tensor<1x120x64x64x2xf16>
func.func @IDFT(%arg0: tensor<1x120x64x64x2xf16>) -> tensor<1x120x64x64x2xf16> {
    %0 = VPU.IDFT(%arg0) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x120x64x64x2xf16> -> tensor<1x120x64x64x2xf16>
    return %0 : tensor<1x120x64x64x2xf16>
    // CHECK: %0 = VPU.Slice %arg0 [0, 0, 0, 0, 0] [1, 60, 64, 64, 2] : tensor<1x120x64x64x2xf16> to tensor<1x60x64x64x2xf16>
    // CHECK: %1 = VPU.IDFT(%0) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x60x64x64x2xf16> -> tensor<1x60x64x64x2xf16>
    // CHECK: %2 = VPU.Slice %arg0 [0, 60, 0, 0, 0] [1, 60, 64, 64, 2] : tensor<1x120x64x64x2xf16> to tensor<1x60x64x64x2xf16>
    // CHECK: %3 = VPU.IDFT(%2) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x60x64x64x2xf16> -> tensor<1x60x64x64x2xf16>
    // CHECK{LITERAL}: %4 = VPU.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0, 0], [0, 60, 0, 0, 0]]} : tensor<1x60x64x64x2xf16>, tensor<1x60x64x64x2xf16> -> tensor<1x120x64x64x2xf16>
    // CHECK: return %4 : tensor<1x120x64x64x2xf16>
}

// -----

// CHECK-LABEL: @RDFT
// CHECK-SAME:  (%arg0: tensor<1x120x64x64xf16>) -> tensor<1x120x64x33x2xf16>
func.func @RDFT(%arg0: tensor<1x120x64x64xf16>) -> tensor<1x120x64x33x2xf16> {
  %1 = VPU.RDFT(%arg0) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x120x64x64xf16> -> tensor<1x120x64x64x2xf16>
  %2 = VPU.Slice %1 [0, 0, 0, 0, 0] [1, 120, 64, 33, 2] : tensor<1x120x64x64x2xf16> to tensor<1x120x64x33x2xf16>
  return %2 : tensor<1x120x64x33x2xf16>

  // CHECK: %0 = VPU.Slice %arg0 [0, 0, 0, 0] [1, 60, 64, 64] : tensor<1x120x64x64xf16> to tensor<1x60x64x64xf16>
  // CHECK:  %1 = VPU.RDFT(%0) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x60x64x64xf16> -> tensor<1x60x64x64x2xf16>
  // CHECK:  %2 = VPU.Slice %arg0 [0, 60, 0, 0] [1, 60, 64, 64] : tensor<1x120x64x64xf16> to tensor<1x60x64x64xf16>
  // CHECK:  %3 = VPU.RDFT(%2) {axes_attr = [2, 3], signal_size_attr = [-1, -1]} : tensor<1x60x64x64xf16> -> tensor<1x60x64x64x2xf16>
  // CHECK{LITERAL}:  %4 = VPU.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0, 0], [0, 60, 0, 0, 0]]} : tensor<1x60x64x64x2xf16>, tensor<1x60x64x64x2xf16> -> tensor<1x120x64x64x2xf16>
  // CHECK:  %5 = VPU.Slice %4 [0, 0, 0, 0, 0] [1, 120, 64, 33, 2] : tensor<1x120x64x64x2xf16> to tensor<1x120x64x33x2xf16>
  // CHECK:  return %5 : tensor<1x120x64x33x2xf16>
}

// -----

// CHECK-LABEL: @IRDFT
// CHECK-SAME: (%arg0: tensor<1x120x64x33x2xf16>) -> tensor<1x120x64x64xf16>
func.func @IRDFT(%arg0: tensor<1x120x64x33x2xf16>) -> tensor<1x120x64x64xf16> {
  %0 = VPU.IRDFT(%arg0) {axes_attr = [3], signal_size_attr = [-1]} : tensor<1x120x64x33x2xf16> -> tensor<1x120x64x64xf16>
  return %0 : tensor<1x120x64x64xf16>

  // CHECK: %0 = VPU.Slice %arg0 [0, 0, 0, 0, 0] [1, 60, 64, 33, 2] : tensor<1x120x64x33x2xf16> to tensor<1x60x64x33x2xf16>
  // CHECK: %1 = VPU.IRDFT(%0) {axes_attr = [3], signal_size_attr = [-1]} : tensor<1x60x64x33x2xf16> -> tensor<1x60x64x64xf16>
  // CHECK: %2 = VPU.Slice %arg0 [0, 60, 0, 0, 0] [1, 60, 64, 33, 2] : tensor<1x120x64x33x2xf16> to tensor<1x60x64x33x2xf16>
  // CHECK: %3 = VPU.IRDFT(%2) {axes_attr = [3], signal_size_attr = [-1]} : tensor<1x60x64x33x2xf16> -> tensor<1x60x64x64xf16>
  // CHECK{LITERAL}: %4 = VPU.Concat(%1, %3) {static_offsets = [[0, 0, 0, 0], [0, 60, 0, 0]]} : tensor<1x60x64x64xf16>, tensor<1x60x64x64xf16> -> tensor<1x120x64x64xf16>
  // CHECK: return %4 : tensor<1x120x64x64xf16>
}
