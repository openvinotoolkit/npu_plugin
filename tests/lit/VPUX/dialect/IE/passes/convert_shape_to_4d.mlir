//
// Copyright (C) 2022-2023 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-shape-to-4d --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK:       func.func @main(
// CHECK-SAME:      [[VAL_0:%.+]]: tensor<1x1000xf32>
// CHECK-SAME:      [[VAL_1:%.+]]: tensor<1x224x224xf32>
// CHECK-SAME:      [[VAL_2:%.+]]: tensor<1x512xf32>
// CHECK-SAME:      [[VAL_3:%.+]]: tensor<8x1024xf32>
func.func @main(%arg0: tensor<1x1000xf32>, %arg1: tensor<1x224x224xf32>, %arg2: tensor<1x512xf32>, %arg3: tensor<8x1024xf32>) ->
        (tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>, tensor<8x1024xf32>) {
    %0 = IE.Clamp(%arg0) {min = 1.0, max = 3.0} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    %1 = IE.Sigmoid(%arg1) : tensor<1x224x224xf32> -> tensor<1x224x224xf32>
    %2 = IE.Elu(%1) {x = 1.0} : tensor<1x224x224xf32> -> tensor<1x224x224xf32>

    %input_low = const.Declare tensor<1x1xf32> = dense<0.0> : tensor<1x1xf32>
    %input_high = const.Declare tensor<1x1xf32> = dense<255.0> : tensor<1x1xf32>
    %output_low = const.Declare tensor<1x1xf32> = dense<0.0> : tensor<1x1xf32>
    %output_high = const.Declare tensor<1x1xf32> = dense<255.0> : tensor<1x1xf32>
    %3 = IE.FakeQuantize(%arg2, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 } :
        tensor<1x512xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<1x512xf32>

    %4 = const.Declare tensor<1xf32> = dense<6.0> : tensor<1xf32>
    %5 = const.Declare tensor<1xf32> = dense<2.0> : tensor<1xf32>
    %6 = IE.Subtract(%arg3, %4) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x1024xf32>, tensor<1xf32> -> tensor<8x1024xf32>
    %7 = IE.Add(%6, %5) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<8x1024xf32>, tensor<1xf32> -> tensor<8x1024xf32>

    return %0, %2, %3, %7 : tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>, tensor<8x1024xf32>

    // CHECK-DAG: [[VAL_4:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG: [[VAL_5:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<6.000000e+00> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG: [[VAL_6:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG: [[VAL_7:%.+]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>]

    // CHECK-DAG:   [[VAL_0_4D:%.+]] = IE.AffineReshape([[VAL_0]]) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf32> -> tensor<1x1x1x1000xf32>
    // CHECK-DAG:   [[VAL_1_4D:%.+]] = IE.AffineReshape([[VAL_1]]) {
    // CHECK-SAME:      shape_value = [1, 1, 224, 224]} : tensor<1x224x224xf32> -> tensor<1x1x224x224xf32>
    // CHECK-DAG:   [[VAL_3_4D:%.+]] = IE.AffineReshape([[VAL_3]]) {
    // CHECK-SAME:      shape_value = [1, 1, 8, 1024]} : tensor<8x1024xf32> -> tensor<1x1x8x1024xf32>

    // CHECK:   [[VAL_8:%.+]] = IE.Clamp([[VAL_0_4D]])
    // CHECK:   [[VAL_8_2D:%.+]] = IE.AffineReshape([[VAL_8]]) {
    // CHECK-SAME:      shape_value = [1, 1000]} : tensor<1x1x1x1000xf32> -> tensor<1x1000xf32>

    // CHECK:   [[VAL_9:%.+]] = IE.Sigmoid([[VAL_1_4D]])
    // CHECK:   [[VAL_10:%.+]] = IE.Elu([[VAL_9]])
    // CHECK-DAG:   [[VAL_10_3D:%.+]] = IE.AffineReshape([[VAL_10]]) {
    // CHECK-SAME:      shape_value = [1, 224, 224]} : tensor<1x1x224x224xf32> -> tensor<1x224x224xf32>
    // CHECK-DAG:   [[VAL_2_4D:%.+]] = IE.AffineReshape([[VAL_2]]) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 512]} : tensor<1x512xf32> -> tensor<1x1x1x512xf32>

    // CHECK:   [[VAL_11:%.+]] = IE.FakeQuantize([[VAL_2_4D]], [[VAL_7]], [[VAL_6]], [[VAL_7]], [[VAL_6]])
    // CHECK:   [[VAL_11_2D:%.+]] = IE.AffineReshape([[VAL_11]]) {
    // CHECK-SAME:      shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>

    // CHECK:   [[VAL_12:%.+]] = IE.Subtract([[VAL_3_4D]], [[VAL_5]])
    // CHECK:   [[VAL_13:%.+]] = IE.Add([[VAL_12]], [[VAL_4]])
    // CHECK:   [[VAL_13_2D:%.+]] = IE.AffineReshape([[VAL_13]]) {
    // CHECK-SAME:      shape_value = [8, 1024]} : tensor<1x1x8x1024xf32> -> tensor<8x1024xf32>

    // CHECK:   return [[VAL_8_2D]], [[VAL_10_3D]], [[VAL_11_2D]], [[VAL_13_2D]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizePerChannel5D(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x3x4x512x64xf32>
func.func @FakeQuantizePerChannel5D(%arg0: tensor<2x3x4x512x64xf32>) -> (tensor<2x3x4x512x64xf32>) {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %output_low = const.Declare tensor<1x1x1x512x1xf32> = dense<10.0> : tensor<1x1x1x512x1xf32>
    %output_high = const.Declare tensor<1x1x1x512x1xf32> = dense<205.0> : tensor<1x1x1x512x1xf32>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32 } :
        tensor<2x3x4x512x64xf32>, tensor<f32>, tensor<f32>, tensor<1x1x1x512x1xf32>, tensor<1x1x1x512x1xf32> -> tensor<2x3x4x512x64xf32>

    return %3 : tensor<2x3x4x512x64xf32>

    // CHECK-DAG: %[[VAL_4:.*]] = const.Declare tensor<1x1x512x1xf32> = dense<2.050000e+02> : tensor<1x1x1x512x1xf32>, [#const.Reshape<[1, 1, 512, 1]>]
    // CHECK-DAG: %[[VAL_3:.*]] = const.Declare tensor<1x1x512x1xf32> = dense<1.000000e+01> : tensor<1x1x1x512x1xf32>, [#const.Reshape<[1, 1, 512, 1]>]
    // CHECK-DAG: %[[VAL_2:.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>

    // CHECK:   %[[RESHAPE_BEFORE:.*]] = IE.Reshape(%[[VAL_0]]) {
    // CHECK-SAME:      shape_value = [1, 24, 512, 64]} : tensor<2x3x4x512x64xf32> -> tensor<1x24x512x64xf32>
    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.*]] = IE.Reshape(%[[FQ]]) {
    // CHECK-SAME:      shape_value = [2, 3, 4, 512, 64]} : tensor<1x24x512x64xf32> -> tensor<2x3x4x512x64xf32>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizePerChannel5DWithDifferentInputOutput(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<2x3x4x512x64xf32>
func.func @FakeQuantizePerChannel5DWithDifferentInputOutput(%arg0: tensor<2x3x4x512x64xf32>) -> (tensor<2x3x4x512x64xf32>) {
    %input_low = const.Declare tensor<1x1x4x1x1xf32> = dense<0.0> : tensor<1x1x4x1x1xf32>
    %input_high = const.Declare tensor<1x1x4x1x1xf32> = dense<255.0> : tensor<1x1x4x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x512x1xf32> = dense<10.0> : tensor<1x1x1x512x1xf32>
    %output_high = const.Declare tensor<1x1x1x512x1xf32> = dense<205.0> : tensor<1x1x1x512x1xf32>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32 } :
        tensor<2x3x4x512x64xf32>, tensor<1x1x4x1x1xf32>, tensor<1x1x4x1x1xf32>, tensor<1x1x1x512x1xf32>, tensor<1x1x1x512x1xf32> -> tensor<2x3x4x512x64xf32>

    return %3 : tensor<2x3x4x512x64xf32>

    // CHECK-DAG: %[[VAL_4:.*]] = const.Declare tensor<1x1x512x1xf32> = dense<2.050000e+02> : tensor<1x1x1x512x1xf32>, [#const.Reshape<[1, 1, 512, 1]>]
    // CHECK-DAG: %[[VAL_3:.*]] = const.Declare tensor<1x1x512x1xf32> = dense<1.000000e+01> : tensor<1x1x1x512x1xf32>, [#const.Reshape<[1, 1, 512, 1]>]
    // CHECK-DAG: %[[VAL_2:.*]] = const.Declare tensor<1x4x1x1xf32> = dense<2.550000e+02> : tensor<1x1x4x1x1xf32>, [#const.Reshape<[1, 4, 1, 1]>]
    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1x4x1x1xf32> = dense<0.000000e+00> : tensor<1x1x4x1x1xf32>, [#const.Reshape<[1, 4, 1, 1]>]

    // CHECK:   %[[RESHAPE_BEFORE:.*]] = IE.AffineReshape(%[[VAL_0]]) {
    // CHECK-SAME:      shape_value = [6, 4, 512, 64]} : tensor<2x3x4x512x64xf32> -> tensor<6x4x512x64xf32>
    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.*]] = IE.AffineReshape(%[[FQ]]) {
    // CHECK-SAME:      shape_value = [2, 3, 4, 512, 64]} : tensor<6x4x512x64xf32> -> tensor<2x3x4x512x64xf32>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizePerChannel3D(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x512x64xf32>
func.func @FakeQuantizePerChannel3D(%arg0: tensor<1x512x64xf32>) -> (tensor<1x512x64xf32>) {
    %input_low = const.Declare tensor<1x512x1xf32> = dense<0.0> : tensor<1x512x1xf32>
    %input_high = const.Declare tensor<1x512x1xf32> = dense<255.0> : tensor<1x512x1xf32>
    %output_low = const.Declare tensor<1x512x1xf32> = dense<10.0> : tensor<1x512x1xf32>
    %output_high = const.Declare tensor<1x512x1xf32> = dense<205.0> : tensor<1x512x1xf32>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32 } :
        tensor<1x512x64xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32>, tensor<1x512x1xf32> -> tensor<1x512x64xf32>

    return %3 : tensor<1x512x64xf32>

    // CHECK-DAG: %[[VAL_4:.*]] = const.Declare tensor<1x512x1x1xf32> = dense<2.050000e+02> : tensor<1x512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]
    // CHECK-DAG: %[[VAL_3:.*]] = const.Declare tensor<1x512x1x1xf32> = dense<1.000000e+01> : tensor<1x512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]
    // CHECK-DAG: %[[VAL_2:.*]] = const.Declare tensor<1x512x1x1xf32> = dense<2.550000e+02> : tensor<1x512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]
    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1x512x1x1xf32> = dense<0.000000e+00> : tensor<1x512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]

    // CHECK:   %[[RESHAPE_BEFORE:.*]] = IE.AffineReshape(%[[VAL_0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 512, 1, 64]} : tensor<1x512x64xf32> -> tensor<1x512x1x64xf32>
    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.*]] = IE.AffineReshape(%[[FQ]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 512, 64]} : tensor<1x512x1x64xf32> -> tensor<1x512x64xf32>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizePerChannel2D(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<512x64xf32>
func.func @FakeQuantizePerChannel2D(%arg0: tensor<512x64xf32>) -> (tensor<512x64xf32>) {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %output_low = const.Declare tensor<512x1xf32> = dense<10.0> : tensor<512x1xf32>
    %output_high = const.Declare tensor<512x1xf32> = dense<205.0> : tensor<512x1xf32>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32 } :
        tensor<512x64xf32>, tensor<f32>, tensor<f32>, tensor<512x1xf32>, tensor<512x1xf32> -> tensor<512x64xf32>

    return %3 : tensor<512x64xf32>

    // CHECK-DAG: %[[VAL_4:.*]] = const.Declare tensor<1x512x1x1xf32> = dense<2.050000e+02> : tensor<512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]
    // CHECK-DAG: %[[VAL_3:.*]] = const.Declare tensor<1x512x1x1xf32> = dense<1.000000e+01> : tensor<512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]
    // CHECK-DAG: %[[VAL_2:.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>

    // CHECK:   %[[RESHAPE_BEFORE:.*]] = IE.AffineReshape(%[[VAL_0]]) {
    // CHECK-SAME:      shape_value = [1, 512, 1, 64]} : tensor<512x64xf32> -> tensor<1x512x1x64xf32>
    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.*]] = IE.AffineReshape(%[[FQ]]) {
    // CHECK-SAME:      shape_value = [512, 64]} : tensor<1x512x1x64xf32> -> tensor<512x64xf32>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizePerTensor(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<512x64xf32>
func.func @FakeQuantizePerTensor(%arg0: tensor<512x64xf32>) -> (tensor<512x64xf32>) {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %output_low = const.Declare tensor<f32> = dense<10.0> : tensor<f32>
    %output_high = const.Declare tensor<f32> = dense<205.0> : tensor<f32>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32 } :
        tensor<512x64xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32> -> tensor<512x64xf32>

    return %3 : tensor<512x64xf32>

    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>
    // CHECK-DAG: %[[VAL_2:.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK-DAG: %[[VAL_3:.*]] = const.Declare tensor<f32> = dense<1.000000e+01> : tensor<f32>
    // CHECK-DAG: %[[VAL_4:.*]] = const.Declare tensor<f32> = dense<2.050000e+02> : tensor<f32>

    // CHECK:   %[[RESHAPE_BEFORE:.*]] = IE.AffineReshape(%[[VAL_0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 512, 64]} : tensor<512x64xf32> -> tensor<1x1x512x64xf32>
    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.*]] = IE.AffineReshape(%[[FQ]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [512, 64]} : tensor<1x1x512x64xf32> -> tensor<512x64xf32>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizePerTensor5Dto4D(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x3x13x13x6xf16>
func.func @FakeQuantizePerTensor5Dto4D(%arg0: tensor<1x3x13x13x6xf16>) -> (tensor<1x3x13x13x6xf16>) {
    %input_low = const.Declare tensor<1x1x1x1x1xf16> = dense<0.950494349> : tensor<1x1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %input_high = const.Declare tensor<1x1x1x1x1xf16> = dense<60.8316383> : tensor<1x1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %output_low = const.Declare tensor<1x1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %output_high = const.Declare tensor<1x1x1x1x1xf16> = dense<62.8316383> : tensor<1x1x1x1x1xf32>, [#const.ConvertElemType<f16>]
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 256 : i32 } :
        tensor<1x3x13x13x6xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16>, tensor<1x1x1x1x1xf16> -> tensor<1x3x13x13x6xf16>

    return %3 : tensor<1x3x13x13x6xf16>

    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.950494349> : tensor<1x1x1x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG: %[[VAL_2:.*]] = const.Declare tensor<1x1x1x1xf16> = dense<60.8316383> : tensor<1x1x1x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG: %[[VAL_3:.*]] = const.Declare tensor<1x1x1x1xf16> = dense<0.000000e+00> : tensor<1x1x1x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG: %[[VAL_4:.*]] = const.Declare tensor<1x1x1x1xf16> = dense<62.8316383> : tensor<1x1x1x1x1xf32>, [#const.ConvertElemType<f16>, #const.Reshape<[1, 1, 1, 1]>]
    // CHECK:   %[[RESHAPE_BEFORE:.*]] = IE.AffineReshape(%[[VAL_0]])
    // CHECK-SAME   {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [3, 13, 13, 6]}
    // CHECK-SAME   tensor<1x3x13x13x6xf16> -> tensor<3x13x13x6xf16>
    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.*]] = IE.AffineReshape(%[[FQ]])
    // CHECK-SAME   {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 3, 13, 13, 6]}
    // CHECK-SAME   tensor<3x13x13x6xf16> -> tensor<1x3x13x13x6xf16>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

// CHECK-LABEL: func.func @FakeQuantizeDifferentInputAndOutput(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<48x3x3x3xf32>
func.func @FakeQuantizeDifferentInputAndOutput(%arg0: tensor<48x3x3x3xf32>) -> (tensor<48x3x3x3xf32>) {
    %input_low = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    %input_high = const.Declare tensor<1xf32> = dense<2.540000e+02> : tensor<1xf32>
    %output_low = const.Declare tensor<48x1x1x1xf32> = dense<-1.000000e+00> : tensor<48x1x1x1xf32>
    %output_high = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>
    %fq = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = #IE.auto_broadcast_type<NUMPY>, levels = 255 : i64} :
        tensor<48x3x3x3xf32>, tensor<1xf32>, tensor<1xf32>, tensor<48x1x1x1xf32>, tensor<48x1x1x1xf32> -> tensor<48x3x3x3xf32>
    return %fq : tensor<48x3x3x3xf32>

    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    // CHECK-DAG: %[[VAL_2:.*]] = const.Declare tensor<1xf32> = dense<2.540000e+02> : tensor<1xf32>
    // CHECK-DAG: %[[VAL_3:.*]] = const.Declare tensor<48x1x1x1xf32> = dense<-1.000000e+00> : tensor<48x1x1x1xf32>
    // CHECK-DAG: %[[VAL_4:.*]] = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>

    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   return %[[FQ]]
}

// -----

func.func @main(%arg0: tensor<1x256x32xf32>) -> tensor<1x256x32xf32> {
    %0 = const.Declare tensor<1x256x1xf32> = dense<6.0> : tensor<1x256x1xf32>
    %1 = IE.ScaleShift(%arg0, %0) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : tensor<1x256x32xf32>, tensor<1x256x1xf32> -> tensor<1x256x32xf32>
    %2 = IE.Clamp(%1) {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x256x32xf32> -> tensor<1x256x32xf32>

    return %2 : tensor<1x256x32xf32>

    // CHECK:       [[VAL_0:%.*]] = const.Declare tensor<1x256x1x1xf32> = dense<6.000000e+00> : tensor<1x256x1xf32>, [
    // CHECK-SAME:      #const.Reshape<[1, 256, 1, 1]>]
    // CHECK:       [[VAL_0_4D:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 256, 1, 32]} : tensor<1x256x32xf32> -> tensor<1x256x1x32xf32>
    // CHECK:       [[VAL_1:%.*]] = IE.ScaleShift([[VAL_0_4D]], [[VAL_0]]) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>}
    // CHECK-SAME:      tensor<1x256x1x32xf32>, tensor<1x256x1x1xf32> -> tensor<1x256x1x32xf32>
    // CHECK:       [[VAL_Reshape:%.*]]  = IE.AffineReshape(%1) {
    // CHECK-SAME:      shape_value = [1, 1, 256, 32]} : tensor<1x256x1x32xf32> -> tensor<1x1x256x32xf32>
    // CHECK:       %[[VAL_2:.*]] = IE.Clamp([[VAL_Reshape]]) {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x1x256x32xf32> -> tensor<1x1x256x32xf32>
    // CHECK:       %[[VAL_1_4D:.*]] = IE.AffineReshape(%[[VAL_2]]) {
    // CHECK-SAME:      shape_value = [1, 256, 32]} : tensor<1x1x256x32xf32> -> tensor<1x256x32xf32>

    // CHECK:   return %[[VAL_1_4D]]
}

// -----

// CHECK-LABEL: func.func @AddOpInput3D
func.func @AddOpInput3D(%arg0: tensor<1x1x64xf16>, %arg1: tensor<1x1x64xf16>) -> tensor<1x1x64xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x64xf16>, tensor<1x1x64xf16> -> tensor<1x1x64xf16>
    return %0 : tensor<1x1x64xf16>

    // CHECK:    %[[Reshape_0:.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 64]} : tensor<1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    %[[Reshape_1:.*]] = IE.AffineReshape(%arg1) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 64]} : tensor<1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    %[[Add:.*]] = IE.Add(%[[Reshape_0]], %[[Reshape_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x64xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    %[[Reshape_out:.*]] = IE.AffineReshape(%[[Add]]) {
    // CHECK-SAME:      shape_value = [1, 1, 64]} : tensor<1x1x1x64xf16> -> tensor<1x1x64xf16>
    // CHECK:    return %[[Reshape_out]]
}

// -----

// CHECK-LABEL: func.func @AddOpInput3DWithBroadcastNoOpt
func.func @AddOpInput3DWithBroadcastNoOpt(%arg0: tensor<1x1x1xf16>, %arg1: tensor<1x1x64xf16>) -> tensor<1x1x64xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1xf16>, tensor<1x1x64xf16> -> tensor<1x1x64xf16>
    return %0 : tensor<1x1x64xf16>

    // CHECK:    %[[Reshape_0:.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:    %[[Reshape_1:.*]] = IE.AffineReshape(%arg1) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 64]} : tensor<1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    %[[Add:.*]] = IE.Add(%[[Reshape_0]], %[[Reshape_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x1x1xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    %[[Reshape_out:.*]] = IE.AffineReshape(%[[Add]]) {
    // CHECK-SAME:      shape_value = [1, 1, 64]} : tensor<1x1x1x64xf16> -> tensor<1x1x64xf16>
    // CHECK:    return %[[Reshape_out]]
}

// -----

// CHECK-LABEL: func.func @AddOpInput2DNoOpt
func.func @AddOpInput2DNoOpt(%arg0: tensor<3x16xf16>, %arg1: tensor<3x16xf16>) -> tensor<3x16xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<3x16xf16>, tensor<3x16xf16> -> tensor<3x16xf16>
    return %0 : tensor<3x16xf16>

    // CHECK:    %[[Reshape_0:.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 3, 16]} : tensor<3x16xf16> -> tensor<1x1x3x16xf16>
    // CHECK:    %[[Reshape_1:.*]] = IE.AffineReshape(%arg1) {
    // CHECK-SAME:      shape_value = [1, 1, 3, 16]} : tensor<3x16xf16> -> tensor<1x1x3x16xf16>
    // CHECK:    %[[Add:.*]] = IE.Add(%[[Reshape_0]], %[[Reshape_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x1x3x16xf16>, tensor<1x1x3x16xf16> -> tensor<1x1x3x16xf16>
    // CHECK:    %[[Reshape_out:.*]] = IE.AffineReshape(%[[Add]]) {
    // CHECK-SAME:      shape_value = [3, 16]} : tensor<1x1x3x16xf16> -> tensor<3x16xf16>
    // CHECK:    return %[[Reshape_out]]
}

// -----

// CHECK-LABEL: @Convert3dAddWithLastDim
func.func @Convert3dAddWithLastDim(%arg0: tensor<1x1x80xf16>) -> tensor<1x1x80xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x1x80xf16> = dense<2.000000e+00> : tensor<1x1x80xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x1x80xf16>, tensor<1x1x80xf16> -> tensor<1x1x80xf16>

    return %ADD : tensor<1x1x80xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x1x1x80xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x1x80xf16>, [#const.Reshape<[1, 1, 1, 80]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 80]
    // CHECK-SAME:  } : tensor<1x1x80xf16> -> tensor<1x1x1x80xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x1x80xf16>, tensor<1x1x1x80xf16> -> tensor<1x1x1x80xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME:      shape_value = [1, 1, 80]
    // CHECK-SAME:  } : tensor<1x1x1x80xf16> -> tensor<1x1x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x1x80xf16>
}

// -----

// CHECK-LABEL: @Convert3dMulWithLastDim
func.func @Convert3dMulWithLastDim(%arg0: tensor<1x1x80xf16>) -> tensor<1x1x80xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x1x80xf16> = dense<2.000000e+00> : tensor<1x1x80xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x1x80xf16>, tensor<1x1x80xf16> -> tensor<1x1x80xf16>

    return %MUL : tensor<1x1x80xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x1x80xf16>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x1x80xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [1, 1, 80]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<1x1x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x1x80xf16>
}

// -----

// CHECK-LABEL: @Convert3dAddWithSecondDim
func.func @Convert3dAddWithSecondDim(%arg0: tensor<1x80x1xf16>) -> tensor<1x80x1xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x80x1xf16> = dense<2.000000e+00> : tensor<1x80x1xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x80x1xf16>, tensor<1x80x1xf16> -> tensor<1x80x1xf16>

    return %ADD : tensor<1x80x1xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x1x80x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x80x1xf16>, [#const.Reshape<[1, 1, 80, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 80, 1]
    // CHECK-SAME:  } : tensor<1x80x1xf16> -> tensor<1x1x80x1xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x80x1xf16>, tensor<1x1x80x1xf16> -> tensor<1x1x80x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME:      shape_value = [1, 80, 1]
    // CHECK-SAME:  } : tensor<1x1x80x1xf16> -> tensor<1x80x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x80x1xf16>
}

// -----

// CHECK-LABEL: @Convert3dMulWithLastDim
func.func @Convert3dMulWithLastDim(%arg0: tensor<1x80x1xf16>) -> tensor<1x80x1xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x80x1xf16> = dense<2.000000e+00> : tensor<1x80x1xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x80x1xf16>, tensor<1x80x1xf16> -> tensor<1x80x1xf16>

    return %MUL : tensor<1x80x1xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x80x1xf16>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [1, 80, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<1x80x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x80x1xf16>
}

// -----

// CHECK-LABEL: @Convert3dAddWithFirstDim
func.func @Convert3dAddWithFirstDim(%arg0: tensor<80x1x1xf16>) -> tensor<80x1x1xf16> {
    %ADD_WEIGHTS = const.Declare tensor<80x1x1xf16> = dense<2.000000e+00> : tensor<80x1x1xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<80x1x1xf16>, tensor<80x1x1xf16> -> tensor<80x1x1xf16>

    return %ADD : tensor<80x1x1xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<80x1x1xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME:      shape_value = [80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<80x1x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<80x1x1xf16>
}

// -----

// CHECK-LABEL: func.func @AddOpInputWith4Dand1D
func.func @AddOpInputWith4Dand1D(%arg0: tensor<1x10x256x256xf16>, %arg1: tensor<1xf16>) -> tensor<1x10x256x256xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1xf16> -> tensor<1x10x256x256xf16>
    return %0 : tensor<1x10x256x256xf16>

    // CHECK:    [[Reshape_0:%.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME:      shape_value = [1, 1, 1, 1]} : tensor<1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:    [[Result:%.*]] = IE.Add(%arg0, [[Reshape_0]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x10x256x256xf16>, tensor<1x1x1x1xf16> -> tensor<1x10x256x256xf16>
    // CHECK:    return [[Result]]
}

// -----

// CHECK-LABEL: @Convert3dMulWithFirstDim
func.func @Convert3dMulWithFirstDim(%arg0: tensor<80x1x1xf16>) -> tensor<80x1x1xf16> {
    %MUL_WEIGHTS = const.Declare tensor<80x1x1xf16> = dense<2.000000e+00> : tensor<80x1x1xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<80x1x1xf16>, tensor<80x1x1xf16> -> tensor<80x1x1xf16>

    return %MUL : tensor<80x1x1xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<80x1x1xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<80x1x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<80x1x1xf16>
}

// -----

// CHECK-LABEL: @Convert3dMulWithDifferentDim
func.func @Convert3dMulWithDifferentDim(%arg0: tensor<1x1x256xf16>) -> tensor<1x256x256xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x256x1xf16> = dense<2.000000e+00> : tensor<1x256x1xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x1x256xf16>, tensor<1x256x1xf16> -> tensor<1x256x256xf16>

    return %MUL : tensor<1x256x256xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x256x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> :  tensor<1x256x1xf16>, [#const.Reshape<[1, 1, 256, 1]>, #const.Reshape<[1, 256, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 256]
    // CHECK-SAME:  } : tensor<1x1x256xf16> -> tensor<1x1x1x256xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x1x256xf16>, tensor<1x256x1x1xf16> -> tensor<1x256x1x256xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [1, 256, 256]
    // CHECK-SAME:  } : tensor<1x256x1x256xf16> -> tensor<1x256x256xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x256x256xf16>
}

// -----

// CHECK-LABEL: @Convert2dAddWithLastDim
func.func @Convert2dAddWithLastDim(%arg0: tensor<1x80xf16>) -> tensor<1x80xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x80xf16> = dense<2.000000e+00> : tensor<1x80xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x80xf16>, tensor<1x80xf16> -> tensor<1x80xf16>

    return %ADD : tensor<1x80xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x1x1x80xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x80xf16>, [#const.Reshape<[1, 1, 1, 80]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 1, 80]
    // CHECK-SAME:  } : tensor<1x80xf16> -> tensor<1x1x1x80xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x1x80xf16>, tensor<1x1x1x80xf16> -> tensor<1x1x1x80xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME:      shape_value = [1, 80]
    // CHECK-SAME:  } : tensor<1x1x1x80xf16> -> tensor<1x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x80xf16>
}

// -----

// CHECK-LABEL: @Convert2dMulWithLastDim
func.func @Convert2dMulWithLastDim(%arg0: tensor<1x80xf16>) -> tensor<1x80xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x80xf16> = dense<2.000000e+00> : tensor<1x80xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<1x80xf16>, tensor<1x80xf16> -> tensor<1x80xf16>

    return %MUL : tensor<1x80xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x80xf16>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [1, 80]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<1x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x80xf16>
}

// -----

// CHECK-LABEL: @Convert2dAddWithFirstDim
func.func @Convert2dAddWithFirstDim(%arg0: tensor<80x1xf16>) -> tensor<80x1xf16> {
    %ADD_WEIGHTS = const.Declare tensor<80x1xf16> = dense<2.000000e+00> : tensor<80x1xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<80x1xf16>, tensor<80x1xf16> -> tensor<80x1xf16>

    return %ADD : tensor<80x1xf16>

    // CHECK-DAG:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x1x80x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<80x1xf16>, [#const.Reshape<[1, 1, 80, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 1, 80, 1]
    // CHECK-SAME:  } : tensor<80x1xf16> -> tensor<1x1x80x1xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x1x80x1xf16>, tensor<1x1x80x1xf16> -> tensor<1x1x80x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME:      shape_value = [80, 1]
    // CHECK-SAME:  } : tensor<1x1x80x1xf16> -> tensor<80x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<80x1xf16>
}

// -----

// CHECK-LABEL: @Convert2dMulWithFirstDim
func.func @Convert2dMulWithFirstDim(%arg0: tensor<80x1xf16>) -> tensor<80x1xf16> {
    %MUL_WEIGHTS = const.Declare tensor<80x1xf16> = dense<2.000000e+00> : tensor<80x1xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<80x1xf16>, tensor<80x1xf16> -> tensor<80x1xf16>

    return %MUL : tensor<80x1xf16>

    // CHECK-DAG:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<80x1xf16>

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<80x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [80, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<80x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<80x1xf16>
}

// -----

// CHECK-LABEL: @Convert3DMulWithFirstDimLargeOne
func.func @Convert3DMulWithFirstDimLargeOne(%arg0: tensor<16x256x32xf32>) -> tensor<16x256x32xf32> {
    %0 = const.Declare tensor<16x256x1xf32> = dense<6.0> : tensor<16x256x1xf32>
    %1 = IE.Multiply(%arg0, %0) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<16x256x32xf32>, tensor<16x256x1xf32> -> tensor<16x256x32xf32>

    return %1 : tensor<16x256x32xf32>

    // CHECK-DAG:   %[[VAL_0:.*]] = const.Declare tensor<1x16x256x1xf32> = dense<6.000000e+00> : tensor<16x256x1xf32>, [#const.Reshape<[1, 16, 256, 1]>]
    // CHECK:       %[[VAL_0_4D:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 16, 256, 32]} : tensor<16x256x32xf32> -> tensor<1x16x256x32xf32>
    // CHECK:       %[[VAL_1:.*]] = IE.Multiply(%0, %cst) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x256x32xf32>, tensor<1x16x256x1xf32> -> tensor<1x16x256x32xf32>
    // CHECK:       %[[VAL_1_4D:.*]] = IE.AffineReshape(%[[VAL_1]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [16, 256, 32]} : tensor<1x16x256x32xf32> -> tensor<16x256x32xf32>

    // CHECK:   return %[[VAL_1_4D]]
}

// -----

// CHECK-LABEL: @Convert3DSubtractWithFirstDimLargeOne
func.func @Convert3DSubtractWithFirstDimLargeOne(%arg0: tensor<64x64x100xf32>, %arg1: tensor<64x1x100xf32>) -> tensor<64x64x100xf32> {
    %1 = IE.Subtract(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<64x64x100xf32>, tensor<64x1x100xf32> -> tensor<64x64x100xf32>

    return %1 : tensor<64x64x100xf32>

    // CHECK:       %[[VAL_0:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 64, 64, 100]} : tensor<64x64x100xf32> -> tensor<1x64x64x100xf32>
    // CHECK:       %[[VAL_1:.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 64, 1, 100]} : tensor<64x1x100xf32> -> tensor<1x64x1x100xf32>
    // CHECK:       %[[SUBSTRACT:.*]] = IE.Subtract(%[[VAL_0]], %[[VAL_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x64x64x100xf32>, tensor<1x64x1x100xf32> -> tensor<1x64x64x100xf32>
    // CHECK:       %[[VAL_2:.*]] = IE.AffineReshape(%[[SUBSTRACT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [64, 64, 100]} : tensor<1x64x64x100xf32> -> tensor<64x64x100xf32>

    // CHECK:   return %[[VAL_2]]
}

// -----

// CHECK-LABEL: @Convert3DAddWithFirstDimLargeOne
func.func @Convert3DAddWithFirstDimLargeOne(%arg0: tensor<16x32x32xf16>) -> tensor<16x32x32xf16> {
    %ADD_WEIGHTS = const.Declare tensor<16x1x1xf16> = dense<2.000000e+00> : tensor<16x1x1xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = #IE.auto_broadcast_type<NUMPY>
    } : tensor<16x32x32xf16>, tensor<16x1x1xf16> -> tensor<16x32x32xf16>

    return %ADD : tensor<16x32x32xf16>

    // CHECK-DAG:       [[VAL_0:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<16x1x1xf16>, [#const.Reshape<[1, 16, 1, 1]>]
    // CHECK:       [[VAL_1:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 16, 32, 32]} : tensor<16x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[ADD:%.*]] = IE.Add([[VAL_1]], [[VAL_0]])
    // CHECK-SAME:      {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x16x32x32xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[VAL_2:%.*]] = IE.AffineReshape([[ADD]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [16, 32, 32]} : tensor<1x16x32x32xf16> -> tensor<16x32x32xf16>

    // CHECK:   return [[VAL_2]]
}

// -----

!qElemType = !quant.uniform<u8:f16, 0.956:128>

// CHECK-LABEL: @Add3dMixPrecision
func.func @Add3dMixPrecision(%arg0: tensor<12x77x64x!qElemType>, %arg1: tensor<12x77x64x!qElemType>) -> tensor<12x77x64x!qElemType> {
    %ADD = IE.Add(%arg0, %arg1) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<12x77x64x!qElemType>, tensor<12x77x64x!qElemType> -> tensor<12x77x64xf16>
    %QUANT = IE.Quantize(%ADD) {dstElemType = !qElemType} : tensor<12x77x64xf16> -> tensor<12x77x64x!qElemType>
    return %QUANT : tensor<12x77x64x!qElemType>

    // CHECK:    [[Reshape_0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 12, 77, 64]} : tensor<12x77x64x!qElemType> -> tensor<1x12x77x64x!qElemType>
    // CHECK:    [[Reshape_1:%.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 12, 77, 64]} : tensor<12x77x64x!qElemType> -> tensor<1x12x77x64x!qElemType>
    // CHECK:    [[Add:%.*]] = IE.Add([[Reshape_0]], [[Reshape_1]]) {auto_broadcast = #IE.auto_broadcast_type<NUMPY>} : tensor<1x12x77x64x!qElemType>, tensor<1x12x77x64x!qElemType> -> tensor<1x12x77x64xf16>
    // CHECK:    [[Reshape_out:%.*]] = IE.AffineReshape([[Add]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2]], shape_value = [12, 77, 64]} : tensor<1x12x77x64xf16> -> tensor<12x77x64xf16>
    // CHECK:    [[Quant:%.*]] = IE.Quantize([[Reshape_out]]) {dstElemType = !qElemType} : tensor<12x77x64xf16> -> tensor<12x77x64x!qElemType>
    // CHECK:    return [[Quant]] : tensor<12x77x64x!qElemType>
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @Convert3DTransposeWithFirstDimLargeOne
func.func @Convert3DTransposeWithFirstDimLargeOne(%arg0: tensor<512x4096x1xf16>) -> tensor<4096x512x1xf16> {
    %0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1, d2) -> (d1, d0, d2)>} : tensor<512x4096x1xf16> -> tensor<4096x512x1xf16>

    return %0 : tensor<4096x512x1xf16>

    // CHECK:       [[VAL_0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 512, 4096, 1]} : tensor<512x4096x1xf16> -> tensor<1x512x4096x1xf16>
    // CHECK:       [[TRANS:%.*]] = IE.Transpose([[VAL_0]])
    // CHECK-SAME:      {order_value = #NHCW} : tensor<1x512x4096x1xf16> -> tensor<1x4096x512x1xf16>
    // CHECK:       [[VAL_1:%.*]] = IE.AffineReshape([[TRANS]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [4096, 512, 1]} : tensor<1x4096x512x1xf16> -> tensor<4096x512x1xf16>

    // CHECK:   return [[VAL_1]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @Convert3DTransposeWithLast
func.func @Convert3DTransposeWithLast(%arg0: tensor<1x512x4096xf16>) -> tensor<1x4096x512xf16> {
    %0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1, d2) -> (d0, d2, d1)>} : tensor<1x512x4096xf16> -> tensor<1x4096x512xf16>

    return %0 : tensor<1x4096x512xf16>

    // CHECK:       [[VAL_0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 1, 512, 4096]} : tensor<1x512x4096xf16> -> tensor<1x1x512x4096xf16>
    // CHECK:       [[TRANS:%.*]] = IE.Transpose([[VAL_0]])
    // CHECK-SAME:      {order_value = #NCWH} : tensor<1x1x512x4096xf16> -> tensor<1x1x4096x512xf16>
    // CHECK:       [[VAL_1:%.*]] = IE.AffineReshape([[TRANS]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [1, 4096, 512]} : tensor<1x1x4096x512xf16> -> tensor<1x4096x512xf16>

    // CHECK:   return [[VAL_1]]
}

// -----

#NCWH = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>

// CHECK-LABEL: @Convert2DTranspose
func.func @Convert2DTranspose(%arg0: tensor<4096x512xf16>) -> tensor<512x4096xf16> {
    %0 = IE.Transpose(%arg0) {order_value = affine_map<(d0, d1) -> (d1, d0)>} : tensor<4096x512xf16> -> tensor<512x4096xf16>

    return %0 : tensor<512x4096xf16>

    // CHECK:       [[VAL_0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 4096, 512]} : tensor<4096x512xf16> -> tensor<1x1x4096x512xf16>
    // CHECK:       [[TRANS:%.*]] = IE.Transpose([[VAL_0]])
    // CHECK-SAME:      {order_value = #NCWH} : tensor<1x1x4096x512xf16> -> tensor<1x1x512x4096xf16>
    // CHECK:       [[VAL_1:%.*]] = IE.AffineReshape([[TRANS]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [0], [1]], shape_value = [512, 4096]} : tensor<1x1x512x4096xf16> -> tensor<512x4096xf16>

    // CHECK:   return [[VAL_1]]
}

// CHECK-LABEL: func.func @ConvertShapeTo4DStridedSlice
func.func @ConvertShapeTo4DStridedSlice(%arg0: tensor<4004x320xf16>) -> (tensor<4004x160xf16>) {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 1], begins_attr = [0, 0], ellipsis_mask = [], end_mask = [1, 0], ends_attr = [4004, 320], new_axis_mask = [], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [], strides_attr = [1, 2]} : tensor<4004x320xf16> -> tensor<4004x160xf16>
    return %0 : tensor<4004x160xf16>

    // CHECK:       [[Reshape_0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1, 2, 3]], shape_value = [4004, 320, 1, 1]} : tensor<4004x320xf16> -> tensor<4004x320x1x1xf16>
    // CHECK:       %[[STRIDEDSLICE:.*]] = IE.StridedSlice(%0) {begin_mask = [0, 1, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0], end_mask = [1, 0, 0, 0], ends_attr = [4004, 320, 1, 1], new_axis_mask = [0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0], strides_attr = [1, 2, 1, 1]} : tensor<4004x320x1x1xf16> -> tensor<4004x160x1x1xf16>
    // CHECK:       %[[Reshape_1:.*]] = IE.AffineReshape(%[[STRIDEDSLICE]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [1], [1], [1]], shape_value = [4004, 160]} : tensor<4004x160x1x1xf16> -> tensor<4004x160xf16>
    // CHECK:    return %[[Reshape_1]]
}

// CHECK-LABEL: func.func @ConvertShapeTo4DFrom5DStridedSlice
func.func @ConvertShapeTo4DFrom5DStridedSlice(%arg0: tensor<1x5x20x32x32xf16>) -> (tensor<1x5x20x32x16xf16>) {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0, 0], begins_attr = [0, 0, 0, 0, 0], ellipsis_mask = [], end_mask = [0, 0, 0, 0, 0], ends_attr = [1, 5, 20, 32, 32], new_axis_mask = [], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1, 2]} : tensor<1x5x20x32x32xf16> -> tensor<1x5x20x32x16xf16>
    return %0 : tensor<1x5x20x32x16xf16>

    // CHECK:       [[Reshape_0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2], [3]], shape_value = [5, 20, 32, 32]} : tensor<1x5x20x32x32xf16> -> tensor<5x20x32x32xf16>
    // CHECK:       %[[STRIDEDSLICE:.*]] = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [5, 20, 32, 32], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<5x20x32x32xf16> -> tensor<5x20x32x16xf16>
    // CHECK:       %[[Reshape_1:.*]] = IE.AffineReshape(%[[STRIDEDSLICE]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3], [4]], shape_value = [1, 5, 20, 32, 16]} : tensor<5x20x32x16xf16> -> tensor<1x5x20x32x16xf16>
    // CHECK:    return %[[Reshape_1]]
}

// CHECK-LABEL: func.func @ConvertShapeTo4DFrom6DStridedSlice
func.func @ConvertShapeTo4DFrom6DStridedSlice(%arg0: tensor<1x1x5x20x32x32xf16>) -> (tensor<1x1x5x20x32x16xf16>) {
    %0 = IE.StridedSlice(%arg0) {begin_mask = [0, 0, 0, 0, 0, 0], begins_attr = [0, 0, 0, 0, 0, 0], ellipsis_mask = [], end_mask = [0, 0, 0, 0, 0, 0], ends_attr = [1, 1, 5, 20, 32, 32], new_axis_mask = [], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [], strides_attr = [1, 1, 1, 1, 1, 2]} : tensor<1x1x5x20x32x32xf16> -> tensor<1x1x5x20x32x16xf16>
    return %0 : tensor<1x1x5x20x32x16xf16>

    // CHECK:       [[Reshape_0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [0], [1], [2], [3]], shape_value = [5, 20, 32, 32]} : tensor<1x1x5x20x32x32xf16> -> tensor<5x20x32x32xf16>
    // CHECK:       %[[STRIDEDSLICE:.*]] = IE.StridedSlice(%0) {begin_mask = [0, 0, 0, 0], begins_attr = [0, 0, 0, 0], ellipsis_mask = [0, 0, 0, 0], end_mask = [0, 0, 0, 0], ends_attr = [5, 20, 32, 32], new_axis_mask = [0, 0, 0, 0], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>, shrink_axis_mask = [0, 0, 0, 0], strides_attr = [1, 1, 1, 2]} : tensor<5x20x32x32xf16> -> tensor<5x20x32x16xf16>
    // CHECK:       %[[Reshape_1:.*]] = IE.AffineReshape(%[[STRIDEDSLICE]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1, 2], [3], [4], [5]], shape_value = [1, 1, 5, 20, 32, 16]} : tensor<5x20x32x16xf16> -> tensor<1x1x5x20x32x16xf16>
    // CHECK:    return %[[Reshape_1]]
}

// -----

// CHECK-LABEL: @Convert2DSoftmax
func.func @Convert2DSoftmax(%arg0: tensor<4096x512xf16>) -> tensor<4096x512xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 1} : tensor<4096x512xf16> -> tensor<4096x512xf16>

    return %0 : tensor<4096x512xf16>

    // CHECK:       [[VAL_0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 4096, 512]} : tensor<4096x512xf16> -> tensor<1x1x4096x512xf16>
    // CHECK:       [[Softmax:%.*]] = IE.SoftMax([[VAL_0]])
    // CHECK-SAME:      {axisInd = 3 : i64} : tensor<1x1x4096x512xf16> -> tensor<1x1x4096x512xf16>
    // CHECK:       [[VAL_1:%.*]] = IE.AffineReshape([[Softmax]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [0], [1]], shape_value = [4096, 512]} : tensor<1x1x4096x512xf16> -> tensor<4096x512xf16>

    // CHECK:   return [[VAL_1]]
}

// -----

// CHECK-LABEL: @Convert3DSoftmax
func.func @Convert3DSoftmax(%arg0: tensor<8x4096x512xf16>) -> tensor<8x4096x512xf16> {
    %0 = IE.SoftMax(%arg0) {axisInd = 2} : tensor<8x4096x512xf16> -> tensor<8x4096x512xf16>

    return %0 : tensor<8x4096x512xf16>

    // CHECK:       [[VAL_0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 8, 4096, 512]} : tensor<8x4096x512xf16> -> tensor<1x8x4096x512xf16>
    // CHECK:       [[SoftMax:%.*]] = IE.SoftMax([[VAL_0]])
    // CHECK-SAME:      {axisInd = 3 : i64} : tensor<1x8x4096x512xf16> -> tensor<1x8x4096x512xf16>
    // CHECK:       [[VAL_1:%.*]] = IE.AffineReshape([[SoftMax]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [8, 4096, 512]} : tensor<1x8x4096x512xf16> -> tensor<8x4096x512xf16>

    // CHECK:   return [[VAL_1]]
}

// -----

// CHECK-LABEL: @Convert3DInterpolate
func.func @Convert3DInterpolate(%arg0: tensor<8x64x1xf16>) -> tensor<8x64x2xf16> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>,
                nearest_mode = <FLOOR>, pads_begin = [0, 0, 0], pads_end = [0, 0, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [0, 1, 2], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
        scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00],
        sizes_attr = [8, 64, 4]} : tensor<8x64x1xf16> -> tensor<8x64x2xf16>

    return %0 : tensor<8x64x2xf16>

    // CHECK: [[INPUT_RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:           {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 8, 64, 1]} : tensor<8x64x1xf16> -> tensor<1x8x64x1xf16>
    // CHECK: [[Interpolate:%.*]] = IE.Interpolate([[INPUT_RESHAPE]])
    // CHECK:   {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:       antialias = false,
    // CHECK-SAME:       pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 0, 0],
    // CHECK:            cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:       axes_attr = [0, 1, 2, 3],
    // CHECK:            operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
    // CHECK:            scales_attr = [1.000000e+00, 1.000000e+00, 1.000000e+00, 2.000000e+00], sizes_attr = [1, 8, 64, 4]} :
    // CHECK:       tensor<1x8x64x1xf16> -> tensor<1x8x64x2xf16>

    // CHECK: [[OUTPUT_RESHAPE:%.*]] = IE.AffineReshape([[Interpolate]])
    // CHECK-SAME{LITERAL}:            {dim_mapping = [[0], [0], [1], [2]], shape_value = [8, 64, 2]} : tensor<1x8x64x2xf16> -> tensor<8x64x2xf16>
    // CHECK: return [[OUTPUT_RESHAPE]] : tensor<8x64x2xf16>

}

// -----

// CHECK-LABEL: @Convert2DInterpolate
func.func @Convert2DInterpolate(%arg0: tensor<8x64xf16>) -> tensor<17x64xf16> {
    %0 = IE.Interpolate(%arg0) {
        attr = #IE.Interpolate<antialias = false, coord_mode = <ASYMMETRIC>, cube_coeff = -7.500000e-01 : f64, mode = <NEAREST>,
                nearest_mode = <FLOOR>, pads_begin = [0, 0], pads_end = [1, 0], shape_calc_mode = <SCALES>>,
        axes_attr = [0, 1], operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
        scales_attr = [2.000000e+00, 1.000000e+00],
        sizes_attr = [8, 64]} : tensor<8x64xf16> -> tensor<17x64xf16>

    return %0 : tensor<17x64xf16>

    // CHECK: [[INPUT_RESHAPE:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:           {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 8, 64]} : tensor<8x64xf16> -> tensor<1x1x8x64xf16>
    // CHECK: [[Interpolate:%.*]] = IE.Interpolate([[INPUT_RESHAPE]])
    // CHECK:   {attr = #IE.Interpolate<mode = <NEAREST>, shape_calc_mode = <SCALES>, coord_mode = <ASYMMETRIC>, nearest_mode = <FLOOR>,
    // CHECK-SAME:       antialias = false,
    // CHECK-SAME:       pads_begin = [0, 0, 0, 0], pads_end = [0, 0, 1, 0],
    // CHECK:            cube_coeff = -7.500000e-01 : f64>,
    // CHECK-SAME:       axes_attr = [0, 1, 2, 3],
    // CHECK:            operand_segment_sizes = dense<[1, 0, 0, 0]> : vector<4xi32>,
    // CHECK:            scales_attr = [1.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00], sizes_attr = [1, 1, 8, 64]} :
    // CHECK:       tensor<1x1x8x64xf16> -> tensor<1x1x17x64xf16>

    // CHECK: [[OUTPUT_RESHAPE:%.*]] = IE.AffineReshape([[Interpolate]])
    // CHECK-SAME{LITERAL}:            {dim_mapping = [[0], [0], [0], [1]], shape_value = [17, 64]} : tensor<1x1x17x64xf16> -> tensor<17x64xf16>
    // CHECK: return [[OUTPUT_RESHAPE]] : tensor<17x64xf16>

}

// -----

// CHECK-LABEL: @ConvertFloor
func.func @ConvertFloor(%arg0: tensor<8x4096x512xf16>) -> tensor<8x4096x512xf16> {
    %0 = IE.Floor(%arg0) : tensor<8x4096x512xf16> -> tensor<8x4096x512xf16>

    return %0 : tensor<8x4096x512xf16>

    // CHECK:       [[VAL_0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 8, 4096, 512]} : tensor<8x4096x512xf16> -> tensor<1x8x4096x512xf16>
    // CHECK:       [[Floor:%.*]] = IE.Floor([[VAL_0]]) : tensor<1x8x4096x512xf16> -> tensor<1x8x4096x512xf16>
    // CHECK:       [[VAL_1:%.*]] = IE.AffineReshape([[Floor]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [8, 4096, 512]} : tensor<1x8x4096x512xf16> -> tensor<8x4096x512xf16>

    // CHECK:   return [[VAL_1]]
}
