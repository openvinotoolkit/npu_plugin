//
// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache 2.0
//
// RUN: vpux-opt --split-input-file --init-compiler="vpu-arch=%arch%" --canonicalize %s | FileCheck %s
// REQUIRES: arch-VPUX30XX || arch-VPUX37XX

// CHECK-LABEL: @FuseScaleAndBias
func @FuseScaleAndBias(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %0 = IE.ScaleShift(%arg0, %weights)
        {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    %bias = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias)
        {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %1 : tensor<1x3x300x300xf32>

    // CHECK-DAG:   %[[WEIGHTS:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   %[[BIAS:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<3.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       %[[VAL0:.*]] = IE.ScaleShift(%arg0, %[[WEIGHTS]], %[[BIAS]])
    // CHECK:       return %[[VAL0]]
}

// -----

// Fuse ScaleShift and Bias should fail
// CHECK-LABEL: @FuseScaleAndBias
func @FuseScaleAndBias(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %bias0 = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %0 = IE.ScaleShift(%arg0, %weights, %bias0)
        {operand_segment_sizes = dense<[1, 1, 1]> : vector<3xi32>}:
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    %bias1 = const.Declare tensor<1x3x1x1xf32> = dense<4.0> : tensor<1x3x1x1xf32>
    %1 = IE.ScaleShift(%0, %bias1)
        {operand_segment_sizes = dense<[1, 0, 1]> : vector<3xi32>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %1 : tensor<1x3x300x300xf32>

    // CHECK-DAG:   %[[VAL0:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   %[[VAL1:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<3.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   %[[VAL2:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<4.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       %[[VAL3:.*]] = IE.ScaleShift(%arg0, %[[VAL0]], %[[VAL1]])
    // CHECK:       %[[VAL4:.*]] = IE.ScaleShift(%[[VAL3]], %[[VAL2]])
    // CHECK:       return %[[VAL4]]
}

// -----

// Fuse Scale and ScaleShift should fail
// CHECK-LABEL: @FuseScaleAndBias
func @FuseScaleAndBias(%arg0: tensor<1x3x300x300xf32>) -> tensor<1x3x300x300xf32> {
    %weights = const.Declare tensor<1x3x1x1xf32> = dense<2.0> : tensor<1x3x1x1xf32>
    %0 = IE.ScaleShift(%arg0, %weights)
        {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>}:
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    %weights1 = const.Declare tensor<1x3x1x1xf32> = dense<3.0> : tensor<1x3x1x1xf32>
    %bias = const.Declare tensor<1x3x1x1xf32> = dense<4.0> : tensor<1x3x1x1xf32>
    %1 = IE.ScaleShift(%0, %weights1, %bias)
        {operand_segment_sizes = dense<[1, 1, 1]> : vector<3xi32>} :
        tensor<1x3x300x300xf32>, tensor<1x3x1x1xf32>, tensor<1x3x1x1xf32> -> tensor<1x3x300x300xf32>

    return %1 : tensor<1x3x300x300xf32>

    // CHECK-DAG:   %[[VAL0:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<2.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   %[[VAL1:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<3.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK-DAG:   %[[VAL2:.*]] = const.Declare tensor<1x3x1x1xf32> = dense<4.000000e+00> : tensor<1x3x1x1xf32>
    // CHECK:       %[[VAL3:.*]] = IE.ScaleShift(%arg0, %[[VAL0]])
    // CHECK:       %[[VAL4:.*]] = IE.ScaleShift(%[[VAL3]], %[[VAL1]], %[[VAL2]])
    // CHECK:       return %[[VAL4]]
}
