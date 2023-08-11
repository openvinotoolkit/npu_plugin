//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-bilinear-to-strided-concat-and-conv
// --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @ConvertBilinearToStridedConcatAndConv_V1
func.func @ConvertBilinearToStridedConcatAndConv_V1(%arg0: tensor<1x20x96x176xf16>) -> tensor<1x20x192x352xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <PYTORCH_HALF_PIXEL>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 352]
         } : tensor<1x20x96x176xf16> -> tensor<1x20x192x352xf16>

    return %0 : tensor<1x20x192x352xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK-DAG: %cst = const.Declare tensor<20x1x2x2xf16> = dense<2.500000e-01> : tensor<20x1x2x2xf16>
    // CHECK: %0 = IE.Concat(%arg0, %arg0) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x20x96x176xf16>, tensor<1x20x96x176xf16> -> tensor<1x20x96x352xf16>
    // CHECK: %1 = IE.Concat(%0, %0) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x20x96x352xf16>, tensor<1x20x96x352xf16> -> tensor<1x20x192x352xf16>
    // CHECK: %2 = IE.Slice %1 [0, 0, 0, 351] [1, 20, 192, 1] : tensor<1x20x192x352xf16> to tensor<1x20x192x1xf16>
    // CHECK{LITERAL}: %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 352]]} : tensor<1x20x192x352xf16>, tensor<1x20x192x1xf16> -> tensor<1x20x192x353xf16>
    // CHECK: %4 = IE.Slice %3 [0, 0, 191, 0] [1, 20, 1, 353] : tensor<1x20x192x353xf16> to tensor<1x20x1x353xf16>
    // CHECK{LITERAL}: %5 = IE.Concat(%3, %4) {static_offsets = [[0, 0, 0, 0], [0, 0, 192, 0]]} : tensor<1x20x192x353xf16>, tensor<1x20x1x353xf16> -> tensor<1x20x193x353xf16>
    // CHECK: %6 = IE.GroupConvolution(%5, %cst) {dilations = [1, 1], groups = 20 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x20x193x353xf16>, tensor<20x1x2x2xf16> -> tensor<1x20x192x352xf16>
    // CHECK: return %6 : tensor<1x20x192x352xf16>

}

// -----

// CHECK-LABEL: @ConvertBilinearToStridedConcatAndConv_V2
func.func @ConvertBilinearToStridedConcatAndConv_V2(%arg0: tensor<1x32x96x176xf16>) -> tensor<1x32x192x352xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 352]
         } : tensor<1x32x96x176xf16> -> tensor<1x32x192x352xf16>

    return %0 : tensor<1x32x192x352xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK-DAG: %cst = const.Declare tensor<32x1x2x2xf16> = dense<2.500000e-01> : tensor<32x1x2x2xf16>
    // CHECK-DAG: %cst_0 = const.Declare tensor<32x1x2x1xf16> = dense<5.000000e-01> : tensor<32x1x2x1xf16>
    // CHECK-DAG: %cst_1 = const.Declare tensor<32x1x1x2xf16> = dense<5.000000e-01> : tensor<32x1x1x2xf16>
    // CHECK: %0 = IE.MaxPool(%arg0) {kernel_size = [1, 1], pads_begin = [0, 0], pads_end = [0, 0], rounding_type = #IE.rounding_type<FLOOR>, strides = [1, 1]} : tensor<1x32x96x176xf16> -> tensor<1x32x96x176xf16>
    // CHECK: %1 = IE.Slice %arg0 [0, 0, 0, 175] [1, 32, 96, 1] : tensor<1x32x96x176xf16> to tensor<1x32x96x1xf16>
    // CHECK{LITERAL}: %2 = IE.Concat(%arg0, %1) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 176]]} : tensor<1x32x96x176xf16>, tensor<1x32x96x1xf16> -> tensor<1x32x96x177xf16>
    // CHECK: %3 = IE.Slice %2 [0, 0, 95, 0] [1, 32, 1, 177] : tensor<1x32x96x177xf16> to tensor<1x32x1x177xf16>
    // CHECK{LITERAL}: %4 = IE.Concat(%2, %3) {static_offsets = [[0, 0, 0, 0], [0, 0, 96, 0]]} : tensor<1x32x96x177xf16>, tensor<1x32x1x177xf16> -> tensor<1x32x97x177xf16>
    // CHECK: %5 = IE.Slice %4 [0, 0, 0, 0] [1, 32, 97, 176] : tensor<1x32x97x177xf16> to tensor<1x32x97x176xf16>
    // CHECK: %6 = IE.GroupConvolution(%2, %cst_1) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x96x177xf16>, tensor<32x1x1x2xf16> -> tensor<1x32x96x176xf16>
    // CHECK: %7 = IE.GroupConvolution(%5, %cst_0) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x97x176xf16>, tensor<32x1x2x1xf16> -> tensor<1x32x96x176xf16>
    // CHECK: %8 = IE.GroupConvolution(%4, %cst) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x97x177xf16>, tensor<32x1x2x2xf16> -> tensor<1x32x96x176xf16>
    // CHECK: %9 = IE.Concat(%0, %6) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x32x96x176xf16>, tensor<1x32x96x176xf16> -> tensor<1x32x96x352xf16>
    // CHECK: %10 = IE.Concat(%7, %8) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x32x96x176xf16>, tensor<1x32x96x176xf16> -> tensor<1x32x96x352xf16>
    // CHECK: %11 = IE.Concat(%9, %10) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x32x96x352xf16>, tensor<1x32x96x352xf16> -> tensor<1x32x192x352xf16>
    // CHECK: return %11 : tensor<1x32x192x352xf16>
}

// -----

// CHECK-LABEL: @ConvertBilinearWithFQToStridedConcatAndConv
func.func @ConvertBilinearWithFQToStridedConcatAndConv(%arg0: tensor<1x32x96x176xf16>) -> tensor<1x32x192x352xf16> {
    %input_low_0 = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high_0 = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %input_low_1 = const.Declare tensor<f32> = dense<10.0> : tensor<f32>
    %input_high_1 = const.Declare tensor<f32> = dense<50.0> : tensor<f32>

    %0 = IE.FakeQuantize(%arg0, %input_low_0, %input_high_0, %input_low_0, %input_high_0)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x32x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x32x96x176xf16>

    %1 = IE.Interpolate(%0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [192, 352]
         } : tensor<1x32x96x176xf16> -> tensor<1x32x192x352xf16>


    %2 = IE.FakeQuantize(%1, %input_low_1, %input_high_1, %input_low_1, %input_high_1)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x32x192x352xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x32x192x352xf16>

    return %2 : tensor<1x32x192x352xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK-DAG:  %cst = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK-DAG:  %cst_0 = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK-DAG:  %cst_1 = const.Declare tensor<f32> = dense<1.000000e+01> : tensor<f32>
    // CHECK-DAG:  %cst_2 = const.Declare tensor<f32> = dense<5.000000e+01> : tensor<f32>
    // CHECK:  [[VAL0:%.*]] = IE.FakeQuantize(%arg0, %cst, %cst_0, %cst, %cst_0) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x32x96x176xf16>
    // CHECK:  [[VAL1:%.*]] = IE.Slice [[VAL0]] [0, 0, 0, 175] [1, 32, 96, 1] : tensor<1x32x96x176xf16> to tensor<1x32x96x1xf16>
    // CHECK:  [[VAL2:%.*]] = IE.Concat([[VAL0]], [[VAL1]]) {per_axis = {axis = 3 : i64}} : tensor<1x32x96x176xf16>, tensor<1x32x96x1xf16> -> tensor<1x32x96x177xf16>
    // CHECK:  [[VAL3:%.*]] = IE.Slice [[VAL2]] [0, 0, 95, 0] [1, 32, 1, 177] : tensor<1x32x96x177xf16> to tensor<1x32x1x177xf16>
    // CHECK:  [[VAL4:%.*]] = IE.Concat([[VAL2]], [[VAL2]]) {per_axis = {axis = 2 : i64}} : tensor<1x32x96x177xf16>, tensor<1x32x1x177xf16> -> tensor<1x32x97x177xf16>
    // CHECK:  [[VAL5:%.*]] = IE.Slice [[VAL4]] [0, 0, 0, 0] [1, 32, 97, 176] : tensor<1x32x97x177xf16> to tensor<1x32x97x176xf16>

    // CHECK:  [[VAL6:%.*]] = IE.FakeQuantize(%cst_3, %cst_4, %cst_5, %cst_4, %cst_6) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x1x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<32x1x1x1xf16>
    // CHECK:  [[VAL7:%.*]] = IE.GroupConvolution([[VAL0]], [[VAL6]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x96x176xf16>, tensor<32x1x1x1xf16> -> tensor<1x32x96x176xf16>
    // CHECK:  [[VAL8:%.*]] = IE.FakeQuantize([[VAL7]], %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x32x96x176xf16>

    // CHECK:  [[VAL9:%.*]] = IE.FakeQuantize(%cst_7, %cst_8, %cst_9, %cst_8, %cst_10) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x1x1x2xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<32x1x1x2xf16>
    // CHECK:  [[VAL10:%.*]] = IE.GroupConvolution([[VAL2]], [[VAL9]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x96x177xf16>, tensor<32x1x1x2xf16> -> tensor<1x32x96x176xf16>
    // CHECK:  [[VAL11:%.*]] = IE.FakeQuantize([[VAL10]], %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x32x96x176xf16>

    // CHECK:  [[VAL12:%.*]] = IE.FakeQuantize(%cst_11, %cst_12, %cst_13, %cst_12, %cst_14) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x1x2x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<32x1x2x1xf16>
    // CHECK:  [[VAL13:%.*]] = IE.GroupConvolution([[VAL5]], [[VAL12]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x97x176xf16>, tensor<32x1x2x1xf16> -> tensor<1x32x96x176xf16>
    // CHECK:  [[VAL14:%.*]] = IE.FakeQuantize([[VAL13]], %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x32x96x176xf16>

    // CHECK:  [[VAL15:%.*]] = IE.FakeQuantize(%cst_15, %cst_16, %cst_17, %cst_16, %cst_18) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x1x2x2xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<32x1x2x2xf16>
    // CHECK:  [[VAL16:%.*]] = IE.GroupConvolution([[VAL4]], [[VAL15]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x97x177xf16>, tensor<32x1x2x2xf16> -> tensor<1x32x96x176xf16>
    // CHECK:  [[VAL17:%.*]] = IE.FakeQuantize([[VAL16]], %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x96x176xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x32x96x176xf16>
    // CHECK:  [[VAL18:%.*]] = IE.Concat([[VAL8]], [[VAL11]]) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x32x96x176xf16>, tensor<1x32x96x176xf16> -> tensor<1x32x96x352xf16>
    // CHECK:  [[VAL19:%.*]] = IE.Concat([[VAL14]], [[VAL17]]) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x32x96x176xf16>, tensor<1x32x96x176xf16> -> tensor<1x32x96x352xf16>

    // CHECK:  [[VAL20:%.*]] = IE.FakeQuantize(%cst_19, %cst_20, %cst_21, %cst_20, %cst_22) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x1x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<32x1x1x1xf16>
    // CHECK:  [[VAL21:%.*]] = IE.GroupConvolution([[VAL18]], [[VAL20]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x96x352xf16>, tensor<32x1x1x1xf16> -> tensor<1x32x96x352xf16>
    // CHECK:  [[VAL22:%.*]] = IE.FakeQuantize([[VAL21]], %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x96x352xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x32x96x352xf16>

    // CHECK:  [[VAL23:%.*]] = IE.FakeQuantize(%cst_23, %cst_24, %cst_25, %cst_24, %cst_26) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} : tensor<32x1x1x1xf16>, tensor<f16>, tensor<f16>, tensor<f16>, tensor<f16> -> tensor<32x1x1x1xf16>
    // CHECK:  [[VAL24:%.*]] = IE.GroupConvolution([[VAL19]], [[VAL23]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x96x352xf16>, tensor<32x1x1x1xf16> -> tensor<1x32x96x352xf16>
    // CHECK:  [[VAL25:%.*]] = IE.FakeQuantize([[VAL24]], %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x96x352xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x32x96x352xf16>
    // CHECK:  [[VAL26:%.*]] = IE.FakeQuantize([[VAL25]], %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x96x352xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x32x96x352xf16>
    // CHECK:  [[VAL27:%.*]] = IE.FakeQuantize([[VAL22]], %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x96x352xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x32x96x352xf16>
    // CHECK:  [[VAL28:%.*]] = IE.Concat([[VAL27]], [[VAL26]]) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x32x96x352xf16>, tensor<1x32x96x352xf16> -> tensor<1x32x192x352xf16>
    // CHECK:  [[VAL29:%.*]] = IE.FakeQuantize([[VAL28]], %cst_1, %cst_2, %cst_1, %cst_2) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i64} : tensor<1x32x192x352xf16>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<1x32x192x352xf16>
    // CHECK:  return [[VAL29]] : tensor<1x32x192x352xf16>
}

// -----

// CHECK-LABEL: @ConvertBilinearToStridedConcatAndConvEnableCMXConcat_V2
func.func @ConvertBilinearToStridedConcatAndConvEnableCMXConcat_V2(%arg0: tensor<1x512x6x11xf16>) -> tensor<1x512x12x22xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <FLOOR>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [2, 3],
         operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [12, 22]
         } : tensor<1x512x6x11xf16> -> tensor<1x512x12x22xf16>

    return %0 : tensor<1x512x12x22xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK-DAG: %cst = const.Declare tensor<512x1x1x1xf16> = dense<1.000000e+00> : tensor<512x1x1x1xf16>
    // CHECK-DAG: %cst_0 = const.Declare tensor<512x1x2x2xf16> = dense<2.500000e-01> : tensor<512x1x2x2xf16>
    // CHECK-DAG: %cst_1 = const.Declare tensor<512x1x2x1xf16> = dense<5.000000e-01> : tensor<512x1x2x1xf16>
    // CHECK-DAG: %cst_2 = const.Declare tensor<512x1x1x2xf16> = dense<5.000000e-01> : tensor<512x1x1x2xf16>
    // CHECK: %0 = IE.Slice %arg0 [0, 0, 0, 10] [1, 512, 6, 1] : tensor<1x512x6x11xf16> to tensor<1x512x6x1xf16>
    // CHECK{LITERAL}: %1 = IE.Concat(%arg0, %0) {static_offsets = [[0, 0, 0, 0], [0, 0, 0, 11]]} : tensor<1x512x6x11xf16>, tenso
    // CHECK: %2 = IE.Slice %1 [0, 0, 5, 0] [1, 512, 1, 12] : tensor<1x512x6x12xf16> to tensor<1x512x1x12xf16>
    // CHECK{LITERAL}: %3 = IE.Concat(%1, %2) {static_offsets = [[0, 0, 0, 0], [0, 0, 6, 0]]} : tensor<1x512x6x12xf16>, tensor<1x
    // CHECK: %4 = IE.Slice %3 [0, 0, 0, 0] [1, 512, 7, 11] : tensor<1x512x7x12xf16> to tensor<1x512x7x11xf16>
    // CHECK: %5 = IE.GroupConvolution(%arg0, %cst) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_e
    // CHECK: %6 = IE.GroupConvolution(%1, %cst_2) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_en
    // CHECK: %7 = IE.GroupConvolution(%4, %cst_1) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_en
    // CHECK: %8 = IE.GroupConvolution(%3, %cst_0) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_en
    // CHECK: %9 = IE.Concat(%5, %6) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x512x6x
    // CHECK: %10 = IE.Concat(%7, %8) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x512x6
    // CHECK: %11 = IE.GroupConvolution(%9, %cst) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_end
    // CHECK: %12 = IE.GroupConvolution(%10, %cst) {dilations = [1, 1], groups = 512 : i64, pads_begin = [0, 0], pads_en
    // CHECK: %13 = IE.Concat(%11, %12) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x512
    // CHECK: return %13 : tensor<1x512x12x22xf16>
}

// -----

// CHECK-LABEL: @ConvertBilinearAlignCornersToStridedConcatAndConv_HW
func.func @ConvertBilinearAlignCornersToStridedConcatAndConv_HW(%arg0: tensor<1x20x96x176xf16>) -> tensor<1x20x191x351xf16> {
    %0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <SIMPLE>,
        pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
        operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [1.9895833730697632, 1.9943181276321411], sizes_attr = [191, 351]
        } : tensor<1x20x96x176xf16> -> tensor<1x20x191x351xf16>

    return %0 : tensor<1x20x191x351xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK-DAG: %cst = const.Declare tensor<20x1x2x2xf16> = dense<2.500000e-01> : tensor<20x1x2x2xf16>
    // CHECK: %0 = IE.Concat(%arg0, %arg0) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x20x96x176xf16>, tensor<1x20x96x176xf16> -> tensor<1x20x96x352xf16>
    // CHECK: %1 = IE.Concat(%0, %0) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x20x96x352xf16>, tensor<1x20x96x352xf16> -> tensor<1x20x192x352xf16>
    // CHECK: %2 = IE.GroupConvolution(%1, %cst) {dilations = [1, 1], groups = 20 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x20x192x352xf16>, tensor<20x1x2x2xf16> -> tensor<1x20x191x351xf16>
    // CHECK: return %2 : tensor<1x20x191x351xf16>
}

// -----

// CHECK-LABEL: @ConvertInterpolate
func.func @ConvertInterpolate(%arg0: tensor<1x256x1x1xf16>) -> tensor<1x256x32x32xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <SIMPLE>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [32.000000e+00, 32.000000e+00], sizes_attr = [32, 32]
         } : tensor<1x256x1x1xf16> -> tensor<1x256x32x32xf16>

    return %0 : tensor<1x256x32x32xf16>

    // CHECK: %0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [3.200000e+01, 3.200000e+01], sizes_attr = [32, 32]} : tensor<1x256x1x1xf16> -> tensor<1x256x32x32xf16>
    // CHECK: return %0 : tensor<1x256x32x32xf16>

}

// -----

// CHECK-LABEL: @NotConvertInterpolateWithChannelNeedAlign
func.func @NotConvertInterpolateWithChannelNeedAlign(%arg0: tensor<1x1x48x80xf16>) -> tensor<1x1x96x160xf16> {
    %0 = IE.Interpolate(%arg0)
         {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <SIMPLE>,
         pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
         operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]
         } : tensor<1x1x48x80xf16> -> tensor<1x1x96x160xf16>

    return %0 : tensor<1x1x96x160xf16>

    // CHECK: %0 = IE.Interpolate(%arg0) {attr =#IE.Interpolate<mode = <LINEAR_ONNX>, shape_calc_mode = <SIZES>, coord_mode = <ASYMMETRIC>, nearest_mode = <SIMPLE>, antialias = false, pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], cube_coeff = -7.500000e-01 : f64>, axes_attr = [2, 3], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.000000e+00, 2.000000e+00], sizes_attr = [96, 160]} : tensor<1x1x48x80xf16> -> tensor<1x1x96x160xf16>
    // CHECK: return %0 : tensor<1x1x96x160xf16>
}

// -----

// CHECK-LABEL: @InterpolateWithAxesNotProvidedFromNgraph
func.func @InterpolateWithAxesNotProvidedFromNgraph(%arg0: tensor<1x256x7x7xf16>) -> tensor<1x256x14x14xf16> {
    %0 = IE.Interpolate(%arg0)
        {attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <ROUND_PREFER_FLOOR>,
        pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SCALES>>, axes_attr = [0, 1, 2, 3],
        operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 2.000000e+00],
        sizes_attr = [1, 256, 14, 14]} : tensor<1x256x7x7xf16> -> tensor<1x256x14x14xf16>

    return %0 : tensor<1x256x14x14xf16>

 // CHECK: [[CST:%.*]] = const.Declare tensor<256x1x1x1xf16> = dense<1.000000e+00> : tensor<256x1x1x1xf16>
 // CHECK: [[CST_0:%.*]] = const.Declare tensor<256x1x2x2xf16> = dense<2.500000e-01> : tensor<256x1x2x2xf16>
 // CHECK: [[CST_1:%.*]] = const.Declare tensor<256x1x2x1xf16> = dense<5.000000e-01> : tensor<256x1x2x1xf16>
 // CHECK: [[CST_2:%.*]] = const.Declare tensor<256x1x1x2xf16> = dense<5.000000e-01> : tensor<256x1x1x2xf16>
 // CHECK: [[VAL0:%.*]] = IE.Slice %arg0 [0, 0, 0, 6] [1, 256, 7, 1] : tensor<1x256x7x7xf16> to tensor<1x256x7x1xf16>
 // CHECK: [[VAL1:%.*]] = IE.Concat(%arg0, [[VAL0]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 0, 7]]} : tensor<1x256x7x7xf16>, tensor<1x256x7x1xf16> -> tensor<1x256x7x8xf16>
 // CHECK: [[VAL2:%.*]] = IE.Slice [[VAL1]] [0, 0, 6, 0] [1, 256, 1, 8] : tensor<1x256x7x8xf16> to tensor<1x256x1x8xf16>
 // CHECK: [[VAL3:%.*]] = IE.Concat([[VAL1]], [[VAL2]]) {static_offsets = {{\[\[}}0, 0, 0, 0], [0, 0, 7, 0]]} : tensor<1x256x7x8xf16>, tensor<1x256x1x8xf16> -> tensor<1x256x8x8xf16>
 // CHECK: [[VAL4:%.*]] = IE.Slice [[VAL3]] [0, 0, 0, 0] [1, 256, 8, 7] : tensor<1x256x8x8xf16> to tensor<1x256x8x7xf16>
 // CHECK: [[VAL5:%.*]] = IE.GroupConvolution(%arg0, [[CST]]) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x7x7xf16>, tensor<256x1x1x1xf16> -> tensor<1x256x7x7xf16>
 // CHECK: [[VAL6:%.*]] = IE.GroupConvolution([[VAL1]], [[CST_2]]) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x7x8xf16>, tensor<256x1x1x2xf16> -> tensor<1x256x7x7xf16>
 // CHECK: [[VAL7:%.*]] = IE.GroupConvolution([[VAL4]], [[CST_1]]) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x8x7xf16>, tensor<256x1x2x1xf16> -> tensor<1x256x7x7xf16>
 // CHECK: [[VAL8:%.*]] = IE.GroupConvolution([[VAL3]], [[CST_0]]) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x8x8xf16>, tensor<256x1x2x2xf16> -> tensor<1x256x7x7xf16>
 // CHECK: [[VAL9:%.*]] = IE.Concat([[VAL5]], [[VAL6]]) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x256x7x7xf16>, tensor<1x256x7x7xf16> -> tensor<1x256x7x14xf16>
 // CHECK: [[VAL10:%.*]] = IE.Concat([[VAL7]], [[VAL8]]) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x256x7x7xf16>, tensor<1x256x7x7xf16> -> tensor<1x256x7x14xf16>
 // CHECK: [[VAL11:%.*]] = IE.GroupConvolution([[VAL9]], [[CST]]) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x7x14xf16>, tensor<256x1x1x1xf16> -> tensor<1x256x7x14xf16>
 // CHECK: [[VAL12:%.*]] = IE.GroupConvolution([[VAL10]], [[CST]]) {dilations = [1, 1], groups = 256 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x256x7x14xf16>, tensor<256x1x1x1xf16> -> tensor<1x256x7x14xf16>
 // CHECK: [[VAL13:%.*]] = IE.Concat([[VAL11]], [[VAL12]]) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 2 : i64}} : tensor<1x256x7x14xf16>, tensor<1x256x7x14xf16> -> tensor<1x256x14x14xf16>
 // CHECK: return [[VAL13]] : tensor<1x256x14x14xf16>
}

// -----

// CHECK-LABEL: @ConvertBilinearAlignCornersToStridedConcatAndConv_V1
func.func @ConvertBilinearAlignCornersToStridedConcatAndConv_V1(%arg0: tensor<1x32x3x3xf16>) -> tensor<1x32x7x7xf16> {
    %0 = IE.Interpolate(%arg0) {attr = #IE.Interpolate<antialias = false, coord_mode = <ALIGN_CORNERS>, cube_coeff = -7.500000e-01 : f64, mode = <LINEAR_ONNX>, nearest_mode = <SIMPLE>,
        pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0], shape_calc_mode = <SIZES>>, axes_attr = [2, 3],
        operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, scales_attr = [2.3333432674407959, 2.3333432674407959], sizes_attr = [7, 7]
        } : tensor<1x32x3x3xf16> -> tensor<1x32x7x7xf16>

    return %0 : tensor<1x32x7x7xf16>

    // CHECK-NOT: IE.Interpolate
    // CHECK-DAG: [[CST:%.*]] = const.Declare tensor<32x1x3x3xf16> = dense<1.110840e-01> : tensor<32x1x3x3xf16>
    // CHECK: [[VAL0:%.*]] = IE.Concat(%arg0, %arg0, %arg0) {per_axis = {axis = 3 : i64, offset = 1 : i64, stride = 3 : i64}} : tensor<1x32x3x3xf16>, tensor<1x32x3x3xf16>, tensor<1x32x3x3xf16> -> tensor<1x32x3x9xf16>
    // CHECK: [[VAL1:%.*]] = IE.Concat([[VAL0]], [[VAL0]], [[VAL0]]) {per_axis = {axis = 2 : i64, offset = 1 : i64, stride = 3 : i64}} : tensor<1x32x3x9xf16>, tensor<1x32x3x9xf16>, tensor<1x32x3x9xf16> -> tensor<1x32x9x9xf16>
    // CHECK: [[VAL2:%.*]] = IE.GroupConvolution([[VAL1]], [[CST]]) {dilations = [1, 1], groups = 32 : i64, pads_begin = [0, 0], pads_end = [0, 0], strides = [1, 1]} : tensor<1x32x9x9xf16>, tensor<32x1x3x3xf16> -> tensor<1x32x7x7xf16>
    // CHECK: return [[VAL2]] : tensor<1x32x7x7xf16>
}
