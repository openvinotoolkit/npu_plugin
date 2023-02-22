//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --convert-shape-to-4d --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK:       func @main(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x1000xf32>
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<1x224x224xf32>
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<1x512xf32>
// CHECK-SAME:      %[[VAL_3:.*]]: tensor<8x1024xf32>
func @main(%arg0: tensor<1x1000xf32>, %arg1: tensor<1x224x224xf32>, %arg2: tensor<1x512xf32>, %arg3: tensor<8x1024xf32>) ->
        (tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>, tensor<8x1024xf32>) {
    %0 = IE.Clamp(%arg0) {min = 1.0, max = 3.0} : tensor<1x1000xf32> -> tensor<1x1000xf32>
    %1 = IE.Sigmoid(%arg1) : tensor<1x224x224xf32> -> tensor<1x224x224xf32>
    %2 = IE.Elu(%1) {x = 1.0} : tensor<1x224x224xf32> -> tensor<1x224x224xf32>

    %input_low = const.Declare tensor<1x1xf32> = dense<0.0> : tensor<1x1xf32>
    %input_high = const.Declare tensor<1x1xf32> = dense<255.0> : tensor<1x1xf32>
    %output_low = const.Declare tensor<1x1xf32> = dense<0.0> : tensor<1x1xf32>
    %output_high = const.Declare tensor<1x1xf32> = dense<255.0> : tensor<1x1xf32>
    %3 = IE.FakeQuantize(%arg2, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
        tensor<1x512xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<1x512xf32>

    %4 = const.Declare tensor<1xf32> = dense<6.0> : tensor<1xf32>
    %5 = const.Declare tensor<1xf32> = dense<2.0> : tensor<1xf32>
    %6 = IE.Subtract(%arg3, %4) {auto_broadcast = "NUMPY"} : tensor<8x1024xf32>, tensor<1xf32> -> tensor<8x1024xf32>
    %7 = IE.Add(%6, %5) {auto_broadcast = "NUMPY"} : tensor<8x1024xf32>, tensor<1xf32> -> tensor<8x1024xf32>

    return %0, %2, %3, %7 : tensor<1x1000xf32>, tensor<1x224x224xf32>, tensor<1x512xf32>, tensor<8x1024xf32>

    // CHECK-DAG: %[[VAL_4:.*]] = const.Declare tensor<1x1x1x1xf32> = dense<2.000000e+00> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG: %[[VAL_5:.*]] = const.Declare tensor<1x1x1x1xf32> = dense<6.000000e+00> : tensor<1xf32>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG: %[[VAL_6:.*]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>]
    // CHECK-DAG: %[[VAL_7:.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1xf32>, [#const.Reshape<[1, 1, 1, 1]>]

    // CHECK:   %[[VAL_0_4D:.*]] = IE.AffineReshape(%[[VAL_0]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 1000]} : tensor<1x1000xf32> -> tensor<1x1x1x1000xf32>
    // CHECK:   %[[VAL_1_4D:.*]] = IE.AffineReshape(%[[VAL_1]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 224, 1, 224]} : tensor<1x224x224xf32> -> tensor<1x224x1x224xf32>
    // CHECK:   %[[VAL_2_4D:.*]] = IE.AffineReshape(%[[VAL_2]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 1, 1, 512]} : tensor<1x512xf32> -> tensor<1x1x1x512xf32>

    // CHECK:   %[[VAL_8:.*]] = IE.Clamp(%[[VAL_0_4D]])
    // CHECK:   %[[VAL_8_2D:.*]] = IE.AffineReshape(%[[VAL_8]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 1000]} : tensor<1x1x1x1000xf32> -> tensor<1x1000xf32>

    // CHECK:   %[[VAL_9:.*]] = IE.Sigmoid(%[[VAL_1_4D]])
    // CHECK:   %[[VAL_10:.*]] = IE.Elu(%[[VAL_9]])
    // CHECK:   %[[VAL_10_3D:.*]] = IE.AffineReshape(%[[VAL_10]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 224, 224]} : tensor<1x224x1x224xf32> -> tensor<1x224x224xf32>

    // CHECK:   %[[VAL_11:.*]] = IE.FakeQuantize(%[[VAL_2_4D]], %[[VAL_7]], %[[VAL_6]], %[[VAL_7]], %[[VAL_6]])
    // CHECK:   %[[VAL_11_2D:.*]] = IE.AffineReshape(%[[VAL_11]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [1, 512]} : tensor<1x1x1x512xf32> -> tensor<1x512xf32>

    // CHECK:   %[[VAL_3_4D:.*]] = IE.AffineReshape(%[[VAL_3]])
    // CHECK-SAME{LITERAL}:  {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 8, 1, 1024]} : tensor<8x1024xf32> -> tensor<1x8x1x1024xf32>
    // CHECK:   %[[VAL_12:.*]] = IE.Subtract(%[[VAL_3_4D]], %[[VAL_5]])
    // CHECK:   %[[VAL_13:.*]] = IE.Add(%[[VAL_12]], %[[VAL_4]])
    // CHECK:   %[[VAL_13_2D:.*]] = IE.AffineReshape(%[[VAL_13:.*]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [8, 1024]} : tensor<1x8x1x1024xf32> -> tensor<8x1024xf32>

    // CHECK:   return %[[VAL_8_2D]], %[[VAL_10_3D]], %[[VAL_11_2D]], %[[VAL_13_2D]]
}

// -----

// CHECK-LABEL: func @FakeQuantizePerChannel3D(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<1x512x64xf32>
func @FakeQuantizePerChannel3D(%arg0: tensor<1x512x64xf32>) -> (tensor<1x512x64xf32>) {
    %input_low = const.Declare tensor<1x512x1xf32> = dense<0.0> : tensor<1x512x1xf32>
    %input_high = const.Declare tensor<1x512x1xf32> = dense<255.0> : tensor<1x512x1xf32>
    %output_low = const.Declare tensor<1x512x1xf32> = dense<10.0> : tensor<1x512x1xf32>
    %output_high = const.Declare tensor<1x512x1xf32> = dense<205.0> : tensor<1x512x1xf32>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 : i32 } :
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

// CHECK-LABEL: func @FakeQuantizePerChannel2D(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<512x64xf32>
func @FakeQuantizePerChannel2D(%arg0: tensor<512x64xf32>) -> (tensor<512x64xf32>) {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %output_low = const.Declare tensor<512x1xf32> = dense<10.0> : tensor<512x1xf32>
    %output_high = const.Declare tensor<512x1xf32> = dense<205.0> : tensor<512x1xf32>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 : i32 } :
        tensor<512x64xf32>, tensor<f32>, tensor<f32>, tensor<512x1xf32>, tensor<512x1xf32> -> tensor<512x64xf32>

    return %3 : tensor<512x64xf32>

    // CHECK-DAG: %[[VAL_4:.*]] = const.Declare tensor<1x512x1x1xf32> = dense<2.050000e+02> : tensor<512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]
    // CHECK-DAG: %[[VAL_3:.*]] = const.Declare tensor<1x512x1x1xf32> = dense<1.000000e+01> : tensor<512x1xf32>, [#const.Reshape<[1, 512, 1, 1]>]
    // CHECK-DAG: %[[VAL_2:.*]] = const.Declare tensor<f32> = dense<2.550000e+02> : tensor<f32>
    // CHECK-DAG: %[[VAL_1:.*]] = const.Declare tensor<f32> = dense<0.000000e+00> : tensor<f32>

    // CHECK:   %[[RESHAPE_BEFORE:.*]] = IE.AffineReshape(%[[VAL_0]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2, 3]], shape_value = [1, 512, 64, 1]} : tensor<512x64xf32> -> tensor<1x512x64x1xf32>
    // CHECK:   %[[FQ:.*]] = IE.FakeQuantize(%[[RESHAPE_BEFORE]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]])
    // CHECK:   %[[RESHAPE_AFTER:.*]] = IE.AffineReshape(%[[FQ]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [1]], shape_value = [512, 64]} : tensor<1x512x64x1xf32> -> tensor<512x64xf32>
    // CHECK:   return %[[RESHAPE_AFTER]]
}

// -----

// CHECK-LABEL: func @FakeQuantizePerTensor(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<512x64xf32>
func @FakeQuantizePerTensor(%arg0: tensor<512x64xf32>) -> (tensor<512x64xf32>) {
    %input_low = const.Declare tensor<f32> = dense<0.0> : tensor<f32>
    %input_high = const.Declare tensor<f32> = dense<255.0> : tensor<f32>
    %output_low = const.Declare tensor<f32> = dense<10.0> : tensor<f32>
    %output_high = const.Declare tensor<f32> = dense<205.0> : tensor<f32>
    %3 = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 : i32 } :
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

// CHECK-LABEL: func @FakeQuantizeDifferentInputAndOutput(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<48x3x3x3xf32>
func @FakeQuantizeDifferentInputAndOutput(%arg0: tensor<48x3x3x3xf32>) -> (tensor<48x3x3x3xf32>) {
    %input_low = const.Declare tensor<1xf32> = dense<0.000000e+00> : tensor<1xf32>
    %input_high = const.Declare tensor<1xf32> = dense<2.540000e+02> : tensor<1xf32>
    %output_low = const.Declare tensor<48x1x1x1xf32> = dense<-1.000000e+00> : tensor<48x1x1x1xf32>
    %output_high = const.Declare tensor<48x1x1x1xf32> = dense<1.000000e+00> : tensor<48x1x1x1xf32>
    %fq = IE.FakeQuantize(%arg0, %input_low, %input_high, %output_low, %output_high)
        {auto_broadcast = "NUMPY", levels = 255 : i64} :
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

func @main(%arg0: tensor<1x256x32xf32>) -> tensor<1x256x32xf32> {
    %0 = const.Declare tensor<1x256x1xf32> = dense<6.0> : tensor<1x256x1xf32>
    %1 = IE.ScaleShift(%arg0, %0) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : tensor<1x256x32xf32>, tensor<1x256x1xf32> -> tensor<1x256x32xf32>
    %2 = IE.Clamp(%1) {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x256x32xf32> -> tensor<1x256x32xf32>

    return %2 : tensor<1x256x32xf32>

    // CHECK-DAG:   %[[VAL_0:.*]] = const.Declare tensor<1x256x1x1xf32> = dense<6.000000e+00> : tensor<1x256x1xf32>, [#const.Reshape<[1, 256, 1, 1]>]
    // CHECK:       %[[VAL_0_4D:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 256, 1, 32]} : tensor<1x256x32xf32> -> tensor<1x256x1x32xf32>
    // CHECK:       %[[VAL_1:.*]] = IE.ScaleShift(%[[VAL_0_4D]], %[[VAL_0]]) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : tensor<1x256x1x32xf32>, tensor<1x256x1x1xf32> -> tensor<1x256x1x32xf32>
    // CHECK:       %[[VAL_2:.*]] = IE.Clamp(%[[VAL_1]]) {max = 1.000000e+00 : f64, min = 0.000000e+00 : f64} : tensor<1x256x1x32xf32> -> tensor<1x256x1x32xf32>
    // CHECK:       %[[VAL_1_4D:.*]] = IE.AffineReshape(%[[VAL_2]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 256, 32]} : tensor<1x256x1x32xf32> -> tensor<1x256x32xf32>

    // CHECK:   return %[[VAL_1_4D]]
}

// -----

// CHECK-LABEL: func @AddOpInput3D
func@AddOpInput3D(%arg0: tensor<1x1x64xf16>, %arg1: tensor<1x1x64xf16>) -> tensor<1x1x64xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x1x64xf16>, tensor<1x1x64xf16> -> tensor<1x1x64xf16>
    return %0 : tensor<1x1x64xf16>

    // CHECK:    %[[Reshape_0:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [1, 64, 1, 1]} : tensor<1x1x64xf16> -> tensor<1x64x1x1xf16>
    // CHECK:    %[[Reshape_1:.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1, 2, 3]], shape_value = [1, 64, 1, 1]} : tensor<1x1x64xf16> -> tensor<1x64x1x1xf16>
    // CHECK:    %[[Add:.*]] = IE.Add(%[[Reshape_0]], %[[Reshape_1]]) {auto_broadcast = "NUMPY"} : tensor<1x64x1x1xf16>, tensor<1x64x1x1xf16> -> tensor<1x64x1x1xf16>
    // CHECK:    %[[Reshape_out:.*]] = IE.AffineReshape(%[[Add]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [2], [2]], shape_value = [1, 1, 64]} : tensor<1x64x1x1xf16> -> tensor<1x1x64xf16>
    // CHECK:    return %[[Reshape_out]]
}

// -----

// CHECK-LABEL: func @AddOpInput3DWithBroadcastNoOpt
func@AddOpInput3DWithBroadcastNoOpt(%arg0: tensor<1x1x1xf16>, %arg1: tensor<1x1x64xf16>) -> tensor<1x1x64xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x1x1xf16>, tensor<1x1x64xf16> -> tensor<1x1x64xf16>
    return %0 : tensor<1x1x64xf16>

    // CHECK:    %[[Reshape_0:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [2, 3]], shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf16> -> tensor<1x1x1x1xf16>
    // CHECK:    %[[Reshape_1:.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1, 2], [3]], shape_value = [1, 1, 1, 64]} : tensor<1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    %[[Add:.*]] = IE.Add(%[[Reshape_0]], %[[Reshape_1]]) {auto_broadcast = "NUMPY"} : tensor<1x1x1x1xf16>, tensor<1x1x1x64xf16> -> tensor<1x1x1x64xf16>
    // CHECK:    %[[Reshape_out:.*]] = IE.AffineReshape(%[[Add]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [1], [1], [2]], shape_value = [1, 1, 64]} : tensor<1x1x1x64xf16> -> tensor<1x1x64xf16>
    // CHECK:    return %[[Reshape_out]]
}

// -----

// CHECK-LABEL: func @AddOpInput2DNoOpt
func@AddOpInput2DNoOpt(%arg0: tensor<3x16xf16>, %arg1: tensor<3x16xf16>) -> tensor<3x16xf16> {
    %0 = IE.Add(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<3x16xf16>, tensor<3x16xf16> -> tensor<3x16xf16>
    return %0 : tensor<3x16xf16>

    // CHECK:    %[[Reshape_0:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 3, 1, 16]} : tensor<3x16xf16> -> tensor<1x3x1x16xf16>
    // CHECK:    %[[Reshape_1:.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1, 2], [3]], shape_value = [1, 3, 1, 16]} : tensor<3x16xf16> -> tensor<1x3x1x16xf16>
    // CHECK:    %[[Add:.*]] = IE.Add(%[[Reshape_0]], %[[Reshape_1]]) {auto_broadcast = "NUMPY"} : tensor<1x3x1x16xf16>, tensor<1x3x1x16xf16> -> tensor<1x3x1x16xf16>
    // CHECK:    %[[Reshape_out:.*]] = IE.AffineReshape(%[[Add]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [0], [1]], shape_value = [3, 16]} : tensor<1x3x1x16xf16> -> tensor<3x16xf16>
    // CHECK:    return %[[Reshape_out]]
}

// -----

// CHECK-LABEL: @Convert3dAddWithLastDim
func @Convert3dAddWithLastDim(%arg0: tensor<1x1x80xf16>) -> tensor<1x1x80xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x1x80xf16> = dense<2.000000e+00> : tensor<1x1x80xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = "NUMPY"
    } : tensor<1x1x80xf16>, tensor<1x1x80xf16> -> tensor<1x1x80xf16>

    return %ADD : tensor<1x1x80xf16>

    // CHECK:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x1x80xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME:      shape_value = [1, 1, 80]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<1x1x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x1x80xf16>
}

// -----

// CHECK-LABEL: @Convert3dMulWithLastDim
func @Convert3dMulWithLastDim(%arg0: tensor<1x1x80xf16>) -> tensor<1x1x80xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x1x80xf16> = dense<2.000000e+00> : tensor<1x1x80xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = "NUMPY"
    } : tensor<1x1x80xf16>, tensor<1x1x80xf16> -> tensor<1x1x80xf16>

    return %MUL : tensor<1x1x80xf16>

    // CHECK:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x1x80xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [1, 1, 80]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<1x1x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x1x80xf16>
}

// -----

// CHECK-LABEL: @Convert3dAddWithSecondDim
func @Convert3dAddWithSecondDim(%arg0: tensor<1x80x1xf16>) -> tensor<1x80x1xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x80x1xf16> = dense<2.000000e+00> : tensor<1x80x1xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = "NUMPY"
    } : tensor<1x80x1xf16>, tensor<1x80x1xf16> -> tensor<1x80x1xf16>

    return %ADD : tensor<1x80x1xf16>

    // CHECK:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x80x1xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME:      shape_value = [1, 80, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<1x80x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x80x1xf16>
}

// -----

// CHECK-LABEL: @Convert3dMulWithLastDim
func @Convert3dMulWithLastDim(%arg0: tensor<1x80x1xf16>) -> tensor<1x80x1xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x80x1xf16> = dense<2.000000e+00> : tensor<1x80x1xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = "NUMPY"
    } : tensor<1x80x1xf16>, tensor<1x80x1xf16> -> tensor<1x80x1xf16>

    return %MUL : tensor<1x80x1xf16>

    // CHECK:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x80x1xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [1, 80, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<1x80x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x80x1xf16>
}

// -----

// CHECK-LABEL: @Convert3dAddWithFirstDim
func @Convert3dAddWithFirstDim(%arg0: tensor<80x1x1xf16>) -> tensor<80x1x1xf16> {
    %ADD_WEIGHTS = const.Declare tensor<80x1x1xf16> = dense<2.000000e+00> : tensor<80x1x1xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = "NUMPY"
    } : tensor<80x1x1xf16>, tensor<80x1x1xf16> -> tensor<80x1x1xf16>

    return %ADD : tensor<80x1x1xf16>

    // CHECK:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<80x1x1xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME:      shape_value = [80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<80x1x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<80x1x1xf16>
}

// -----

// CHECK-LABEL: @Convert3dMulWithFirstDim
func @Convert3dMulWithFirstDim(%arg0: tensor<80x1x1xf16>) -> tensor<80x1x1xf16> {
    %MUL_WEIGHTS = const.Declare tensor<80x1x1xf16> = dense<2.000000e+00> : tensor<80x1x1xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = "NUMPY"
    } : tensor<80x1x1xf16>, tensor<80x1x1xf16> -> tensor<80x1x1xf16>

    return %MUL : tensor<80x1x1xf16>

    // CHECK:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<80x1x1xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<80x1x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<80x1x1xf16>
}

// -----

// CHECK-LABEL: @Convert2dAddWithLastDim
func @Convert2dAddWithLastDim(%arg0: tensor<1x80xf16>) -> tensor<1x80xf16> {
    %ADD_WEIGHTS = const.Declare tensor<1x80xf16> = dense<2.000000e+00> : tensor<1x80xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = "NUMPY"
    } : tensor<1x80xf16>, tensor<1x80xf16> -> tensor<1x80xf16>

    return %ADD : tensor<1x80xf16>

    // CHECK:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME:      shape_value = [1, 80]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<1x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x80xf16>
}

// -----

// CHECK-LABEL: @Convert2dMulWithLastDim
func @Convert2dMulWithLastDim(%arg0: tensor<1x80xf16>) -> tensor<1x80xf16> {
    %MUL_WEIGHTS = const.Declare tensor<1x80xf16> = dense<2.000000e+00> : tensor<1x80xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = "NUMPY"
    } : tensor<1x80xf16>, tensor<1x80xf16> -> tensor<1x80xf16>

    return %MUL : tensor<1x80xf16>

    // CHECK:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<1x80xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<1x80xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [1, 80]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<1x80xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<1x80xf16>
}

// -----

// CHECK-LABEL: @Convert2dAddWithFirstDim
func @Convert2dAddWithFirstDim(%arg0: tensor<80x1xf16>) -> tensor<80x1xf16> {
    %ADD_WEIGHTS = const.Declare tensor<80x1xf16> = dense<2.000000e+00> : tensor<80x1xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = "NUMPY"
    } : tensor<80x1xf16>, tensor<80x1xf16> -> tensor<80x1xf16>

    return %ADD : tensor<80x1xf16>

    // CHECK:   [[ADD_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<80x1xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<80x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[ADD:%.*]] = IE.Add([[RESHAPE_INPUT]], [[ADD_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[ADD]]) {
    // CHECK-SAME:      shape_value = [80, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<80x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<80x1xf16>
}

// -----

// CHECK-LABEL: @Convert2dMulWithFirstDim
func @Convert2dMulWithFirstDim(%arg0: tensor<80x1xf16>) -> tensor<80x1xf16> {
    %MUL_WEIGHTS = const.Declare tensor<80x1xf16> = dense<2.000000e+00> : tensor<80x1xf16>

    %MUL = IE.Multiply(%arg0, %MUL_WEIGHTS) {
        auto_broadcast = "NUMPY"
    } : tensor<80x1xf16>, tensor<80x1xf16> -> tensor<80x1xf16>

    return %MUL : tensor<80x1xf16>

    // CHECK:   [[MUL_WEIGHTS:%.*]] = const.Declare tensor<1x80x1x1xf16> =
    // CHECK-SAME:  dense<2.000000e+00> : tensor<80x1xf16>, [#const.Reshape<[1, 80, 1, 1]>]

    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SAME:      shape_value = [1, 80, 1, 1]
    // CHECK-SAME:  } : tensor<80x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[MUL:%.*]] = IE.Multiply([[RESHAPE_INPUT]], [[MUL_WEIGHTS]]) {
    // CHECK-SAME:      auto_broadcast = "NUMPY"
    // CHECK-SAME:  } : tensor<1x80x1x1xf16>, tensor<1x80x1x1xf16> -> tensor<1x80x1x1xf16>

    // CHECK:   [[RESHAPE_OUT:%.*]] = IE.AffineReshape([[MUL]]) {
    // CHECK-SAME:      shape_value = [80, 1]
    // CHECK-SAME:  } : tensor<1x80x1x1xf16> -> tensor<80x1xf16>

    // CHECK:   return [[RESHAPE_OUT]] : tensor<80x1xf16>
}

// -----

// CHECK-LABEL: @Convert3DMulWithFirstDimLargeOne
func @Convert3DMulWithFirstDimLargeOne(%arg0: tensor<16x256x32xf32>) -> tensor<16x256x32xf32> {
    %0 = const.Declare tensor<16x256x1xf32> = dense<6.0> : tensor<16x256x1xf32>
    %1 = IE.Multiply(%arg0, %0) {
        auto_broadcast = "NUMPY"
    } : tensor<16x256x32xf32>, tensor<16x256x1xf32> -> tensor<16x256x32xf32>

    return %1 : tensor<16x256x32xf32>

    // CHECK-DAG:   %[[VAL_0:.*]] = const.Declare tensor<1x16x256x1xf32> = dense<6.000000e+00> : tensor<16x256x1xf32>, [#const.Reshape<[1, 16, 256, 1]>]
    // CHECK:       %[[VAL_0_4D:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 16, 256, 32]} : tensor<16x256x32xf32> -> tensor<1x16x256x32xf32>
    // CHECK:       %[[VAL_1:.*]] = IE.Multiply(%0, %cst) {auto_broadcast = "NUMPY"} : tensor<1x16x256x32xf32>, tensor<1x16x256x1xf32> -> tensor<1x16x256x32xf32>
    // CHECK:       %[[VAL_1_4D:.*]] = IE.AffineReshape(%[[VAL_1]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [16, 256, 32]} : tensor<1x16x256x32xf32> -> tensor<16x256x32xf32>

    // CHECK:   return %[[VAL_1_4D]]
}

// -----

// CHECK-LABEL: @Convert3DSubtractWithFirstDimLargeOne
func @Convert3DSubtractWithFirstDimLargeOne(%arg0: tensor<64x64x100xf32>, %arg1: tensor<64x1x100xf32>) -> tensor<64x64x100xf32> {
    %1 = IE.Subtract(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<64x64x100xf32>, tensor<64x1x100xf32> -> tensor<64x64x100xf32>

    return %1 : tensor<64x64x100xf32>

    // CHECK:       %[[VAL_0:.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 64, 64, 100]} : tensor<64x64x100xf32> -> tensor<1x64x64x100xf32>
    // CHECK:       %[[VAL_1:.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 64, 1, 100]} : tensor<64x1x100xf32> -> tensor<1x64x1x100xf32>
    // CHECK:       %[[SUBSTRACT:.*]] = IE.Subtract(%[[VAL_0]], %[[VAL_1]]) {auto_broadcast = "NUMPY"} : tensor<1x64x64x100xf32>, tensor<1x64x1x100xf32> -> tensor<1x64x64x100xf32>
    // CHECK:       %[[VAL_2:.*]] = IE.AffineReshape(%[[SUBSTRACT]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [64, 64, 100]} : tensor<1x64x64x100xf32> -> tensor<64x64x100xf32>

    // CHECK:   return %[[VAL_2]]
}

// -----

// CHECK-LABEL: @Convert3DAddWithFirstDimLargeOne
func @Convert3DAddWithFirstDimLargeOne(%arg0: tensor<16x32x32xf16>) -> tensor<16x32x32xf16> {
    %ADD_WEIGHTS = const.Declare tensor<16x1x1xf16> = dense<2.000000e+00> : tensor<16x1x1xf16>

    %ADD = IE.Add(%arg0, %ADD_WEIGHTS) {
        auto_broadcast = "NUMPY"
    } : tensor<16x32x32xf16>, tensor<16x1x1xf16> -> tensor<16x32x32xf16>

    return %ADD : tensor<16x32x32xf16>

    // CHECK:       [[VAL_0:%.*]] = const.Declare tensor<1x16x1x1xf16> = dense<2.000000e+00> : tensor<16x1x1xf16>, [#const.Reshape<[1, 16, 1, 1]>]
    // CHECK:       [[VAL_1:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 16, 32, 32]} : tensor<16x32x32xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[ADD:%.*]] = IE.Add([[VAL_1]], [[VAL_0]])
    // CHECK-SAME:      {auto_broadcast = "NUMPY"} : tensor<1x16x32x32xf16>, tensor<1x16x1x1xf16> -> tensor<1x16x32x32xf16>
    // CHECK:       [[VAL_2:%.*]] = IE.AffineReshape([[ADD]])
    // CHECK-SAME{LITERAL}:      {dim_mapping = [[0], [0], [1], [2]], shape_value = [16, 32, 32]} : tensor<1x16x32x32xf16> -> tensor<16x32x32xf16>

    // CHECK:   return [[VAL_2]]
}

// -----

!qElemType = type !quant.uniform<u8:f16, 0.956:128>

// CHECK-LABEL: @Add3dMixPrecision
func @Add3dMixPrecision(%arg0: tensor<12x77x64x!qElemType>, %arg1: tensor<12x77x64x!qElemType>) -> tensor<12x77x64x!qElemType> {
    %ADD = IE.Add(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<12x77x64x!qElemType>, tensor<12x77x64x!qElemType> -> tensor<12x77x64xf16>
    %QUANT = IE.Quantize(%ADD) {dstElemType = !qElemType} : tensor<12x77x64xf16> -> tensor<12x77x64x!qElemType>
    return %QUANT : tensor<12x77x64x!qElemType>

    // CHECK:    [[Reshape_0:%.*]] = IE.AffineReshape(%arg0)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 12, 77, 64]} : tensor<12x77x64x!qElemType> -> tensor<1x12x77x64x!qElemType>
    // CHECK:    [[Reshape_1:%.*]] = IE.AffineReshape(%arg1)
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0, 1], [2], [3]], shape_value = [1, 12, 77, 64]} : tensor<12x77x64x!qElemType> -> tensor<1x12x77x64x!qElemType>
    // CHECK:    [[Add:%.*]] = IE.Add([[Reshape_0]], [[Reshape_1]]) {auto_broadcast = "NUMPY"} : tensor<1x12x77x64x!qElemType>, tensor<1x12x77x64x!qElemType> -> tensor<1x12x77x64xf16>
    // CHECK:    [[Reshape_out:%.*]] = IE.AffineReshape([[Add]])
    // CHECK-SAME{LITERAL}: {dim_mapping = [[0], [0], [1], [2]], shape_value = [12, 77, 64]} : tensor<1x12x77x64xf16> -> tensor<12x77x64xf16>
    // CHECK:    [[Quant:%.*]] = IE.Quantize([[Reshape_out]]) {dstElemType = !qElemType} : tensor<12x77x64xf16> -> tensor<12x77x64x!qElemType>
    // CHECK:    return [[Quant]] : tensor<12x77x64x!qElemType>
}

// -----

// CHECK-LABEL: @Convert2dTopKPositiveAxis
func @Convert2dTopKPositiveAxis(%arg0: tensor<80x77xsi32>) -> (tensor<80x1xsi32>, tensor<80x1xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = "MAX", sort = "NONE"} :
                                            tensor<80x77xsi32>, tensor<si32> -> tensor<80x1xsi32>, tensor<80x1xsi32>

    return %output_values, %target_shape : tensor<80x1xsi32>, tensor<80x1xsi32>

    // CHECK:   [[CST_K:%.*]] = const.Declare tensor<si32> = dense<1> : tensor<si32>
    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SMAE:         shape_value = [1, 80, 1, 77]
    // CHECK-SAME:     } : tensor<80x77xsi32> -> tensor<1x80x1x77xsi32>

    // CHECK:   [[VALUE:%.*]], [[SHAPE:%.*]] = IE.TopK([[RESHAPE_INPUT]], [[CST_K]])
    // CHECK-SAME:         {axis = 3 : i64, element_type = si32, mode = "MAX", sort = "NONE"} :
    // CHECK-SAME:         tensor<1x80x1x77xsi32>, tensor<si32> -> tensor<1x80x1x1xsi32>, tensor<1x80x1x1xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.*]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:                    } : tensor<1x80x1x1xsi32> -> tensor<80x1xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.*]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:                    } : tensor<1x80x1x1xsi32> -> tensor<80x1xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]

}

// -----

// CHECK-LABEL: @Convert2dTopKNegativeAxis
func @Convert2dTopKNegativeAxis(%arg0: tensor<80x77xsi32>) -> (tensor<1x77xsi32>, tensor<1x77xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = -2 : i64, element_type = si32, mode = "MAX", sort = "NONE"} :
                                            tensor<80x77xsi32>, tensor<si32> -> tensor<1x77xsi32>, tensor<1x77xsi32>

    return %output_values, %target_shape : tensor<1x77xsi32>, tensor<1x77xsi32>

    // CHECK:   [[CST_K:%.*]] = const.Declare tensor<si32> = dense<1> : tensor<si32>
    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SMAE:         shape_value = [1, 80, 1, 77]
    // CHECK-SAME:     } : tensor<80x77xsi32> -> tensor<1x80x1x77xsi32>

    // CHECK:   [[VALUE:%.*]], [[SHAPE:%.*]] = IE.TopK([[RESHAPE_INPUT]], [[CST_K]])
    // CHECK-SAME:         {axis = 1 : i64, element_type = si32, mode = "MAX", sort = "NONE"} :
    // CHECK-SAME:         tensor<1x80x1x77xsi32>, tensor<si32> -> tensor<1x1x1x77xsi32>, tensor<1x1x1x77xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.*]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:                    } : tensor<1x1x1x77xsi32> -> tensor<1x77xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.*]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:                    } : tensor<1x1x1x77xsi32> -> tensor<1x77xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]
}

// -----

// CHECK-LABEL: @Convert3dTopKPositiveAxis
func @Convert3dTopKPositiveAxis(%arg0: tensor<60x80x77xsi32>) -> (tensor<60x1x77xsi32>, tensor<60x1x77xsi32>) {
    %cst_K = const.Declare tensor<si32> = dense<1> : tensor<si32>
    %output_values, %target_shape = IE.TopK(%arg0, %cst_K) {axis = 1 : i64, element_type = si32, mode = "MAX", sort = "NONE"} :
                                            tensor<60x80x77xsi32>, tensor<si32> -> tensor<60x1x77xsi32>, tensor<60x1x77xsi32>

    return %output_values, %target_shape : tensor<60x1x77xsi32>, tensor<60x1x77xsi32>

    // CHECK:   [[CST_K:%.*]] = const.Declare tensor<si32> = dense<1> : tensor<si32>
    // CHECK:   [[RESHAPE_INPUT:%.*]] = IE.AffineReshape(%arg0) {
    // CHECK-SMAE:         shape_value = [60, 80, 1, 77]
    // CHECK-SAME:     } : tensor<60x80x77xsi32> -> tensor<60x80x1x77xsi32>

    // CHECK:   [[VALUE:%.*]], [[SHAPE:%.*]] = IE.TopK([[RESHAPE_INPUT]], [[CST_K]])
    // CHECK-SAME:         {axis = 1 : i64, element_type = si32, mode = "MAX", sort = "NONE"} :
    // CHECK-SAME:         tensor<60x80x1x77xsi32>, tensor<si32> -> tensor<60x1x1x77xsi32>, tensor<60x1x1x77xsi32>

    // CHECK:   [[RESHAPE_VALUE:%.*]] = IE.AffineReshape([[VALUE]]) {
    // CHECK-SAME:                    } : tensor<60x1x1x77xsi32> -> tensor<60x1x77xsi32>
    // CHECK:   [[RESHAPE_SHAPE:%.*]] = IE.AffineReshape([[SHAPE]]) {
    // CHECK-SAME:                    } : tensor<60x1x1x77xsi32> -> tensor<60x1x77xsi32>
    // CHECK:   return [[RESHAPE_VALUE]], [[RESHAPE_SHAPE]]
}

// -----

#NHCW = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>

// CHECK-LABEL: @Convert3DTransposeWithFirstDimLargeOne
func @Convert3DTransposeWithFirstDimLargeOne(%arg0: tensor<512x4096x1xf16>) -> tensor<4096x512x1xf16> {
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
func @Convert3DTransposeWithLast(%arg0: tensor<1x512x4096xf16>) -> tensor<1x4096x512xf16> {
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
func @Convert2DTranspose(%arg0: tensor<4096x512xf16>) -> tensor<512x4096xf16> {
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
