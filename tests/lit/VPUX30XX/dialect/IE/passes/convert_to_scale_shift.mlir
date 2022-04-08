//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=VPUX30XX" --convert-to-scale-shift %s | FileCheck %s

// CHECK-LABEL: @ConvertAddToScaleShift
func @ConvertAddToScaleShift(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %bias = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %0 = IE.Add(%arg0, %bias)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[BIAS]]) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertAddToScaleShiftBroadcastChannels
func @ConvertAddToScaleShiftBroadcastChannels(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %bias = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %0 = IE.Add(%arg0, %bias)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x300x300xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 3 : i64>]
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[BIAS]]) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertAddWithConstFQToScaleShiftBroadcastChannels
func @ConvertAddWithConstFQToScaleShiftBroadcastChannels(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %bias = const.Declare tensor<1x1x1x1xf32> = dense<2.0> : tensor<1x1x1x1xf32>
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%bias, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
        tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>

    %1 = IE.Multiply(%arg0, %0)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x300x300xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x300x300xf32>

    return %1 : tensor<1x3x300x300xf32>

    // CHECK:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 3 : i64>]
    // CHECK:       %[[CONST_HIGH:.*]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK:       %[[CONST_LOW:.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.FakeQuantize(%[[BIAS]], %[[CONST_LOW]], %[[CONST_HIGH]], %[[CONST_LOW]], %[[CONST_HIGH]]) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x3x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x1x1xf32>
    // CHECK:       %[[VAL1:.*]] = IE.ScaleShift(%arg0, %[[VAL0]]) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL1]]
}

// -----

// CHECK-LABEL: @ConvertAddToScaleShiftReversedInputs
func @ConvertAddToScaleShiftReversedInputs(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %bias = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %0 = IE.Add(%bias, %arg0)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x1x1xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK:       %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[BIAS]]) {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @cannotConvertAddToScaleShift
func @cannotConvertAddToScaleShift(%arg0: tensor<1x3x1x1xf32>, %arg1: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %0 = IE.Add(%arg0, %arg1)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x1x1xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK:       %[[VAL0:.*]] = IE.Add(%arg0, %arg1) {auto_broadcast = "NUMPY"} : tensor<1x3x1x1xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShift
func @ConvertMultiplyToScaleShift(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK:       %[[WEIGHTS:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<3.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[WEIGHTS]]) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShiftBroadcastChannels
func @ConvertMultiplyToScaleShiftBroadcastChannels(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x1x1x1xf32> = dense<3.0> : tensor<1x1x1x1xf32>
    %0 = IE.Multiply(%arg0, %weights)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x300x300xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK:       %[[WEIGHTS:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<3.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 3 : i64>]
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[WEIGHTS]]) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyToScaleShiftReversedInputs
func @ConvertMultiplyToScaleShiftReversedInputs(%arg0: tensor<1x3x1x1xf32>, %arg1: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %0 = IE.Multiply(%arg0, %arg1)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x1x1xf32>, tensor<1x3x300x300xf32> -> tensor<1x3x300x300xf32>

    return %0 : tensor<1x3x300x300xf32>

    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg1, %arg0) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL0]]
}

// -----

// CHECK-LABEL: @ConvertMultiplyWithConstFQToScaleShiftBroadcastChannels
func @ConvertMultiplyWithConstFQToScaleShiftBroadcastChannels(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x1x1x1xf32> = dense<3.0> : tensor<1x1x1x1xf32>
    %input_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %input_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>
    %output_low = const.Declare tensor<1x1x1x1xf32> = dense<0.0> : tensor<1x1x1x1xf32>
    %output_high = const.Declare tensor<1x1x1x1xf32> = dense<255.0> : tensor<1x1x1x1xf32>

    %0 = IE.FakeQuantize(%weights, %input_low, %input_high, %output_low, %output_high)
        { auto_broadcast = "NUMPY", levels = 256 } :
        tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x1x1x1xf32>

    %1 = IE.Reshape(%0) { shape_value = [1, 1, 1] } : tensor<1x1x1x1xf32> -> tensor<1x1x1xf32>
    %2 = IE.Reshape(%1) { shape_value = [1, 1, 1, 1] } : tensor<1x1x1xf32> -> tensor<1x1x1x1xf32>

    %3 = IE.Multiply(%arg0, %2)
        { auto_broadcast = "NUMPY" } :
        tensor<1x3x300x300xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x300x300xf32>

    return %3 : tensor<1x3x300x300xf32>

    // CHECK:       %[[WEIGHTS:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<3.000000e+00> : tensor<1x1x1x1xf32>, [#const.Broadcast<1 : i64, 3 : i64>]
    // CHECK:       %[[CONST_HIGH:.*]] = const.Declare tensor<1x1x1x1xf32> = dense<2.550000e+02> : tensor<1x1x1x1xf32>
    // CHECK:       %[[CONST_LOW:.*]] = const.Declare tensor<1x1x1x1xf32> = dense<0.000000e+00> : tensor<1x1x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.FakeQuantize(%[[WEIGHTS]], %[[CONST_LOW]], %[[CONST_HIGH]], %[[CONST_LOW]], %[[CONST_HIGH]]) {auto_broadcast = "NUMPY", levels = 256 : i64} : tensor<1x3x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x1x1xf32>
    // CHECK:       %[[VAL1:.*]] = IE.Reshape(%[[VAL0]]) {shape_value = [1, 1, 1]} : tensor<1x3x1x1xf32> -> tensor<1x1x1xf32>
    // CHECK:       %[[VAL2:.*]] = IE.Reshape(%[[VAL1]]) {shape_value = [1, 1, 1, 1]} : tensor<1x1x1xf32> -> tensor<1x1x1x1xf32>
    // CHECK:       %[[VAL3:.*]] = IE.ScaleShift(%arg0, %[[VAL2]]) {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : tensor<1x3x300x300xf32>, tensor<1x1x1x1xf32> -> tensor<1x3x300x300xf32>
    // CHECK:       return %[[VAL3]]
}

